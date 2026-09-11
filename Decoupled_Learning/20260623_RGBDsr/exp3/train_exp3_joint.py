"""Exp3 joint origin+RF training for ConvGuided, DKN, and DCTNet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
try:
    import h5py  # optional: online DCTNet mode does not require it
except ImportError:
    h5py = None
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp1.convguided_model import SurveyDGF  # noqa: E402
from exp1.nyuv2_full_dataset import NYUv2FullImageDataset  # noqa: E402
from thirdparty_refine_models import Exp6DRefine3Stage, _pil_bicubic_upsample, _prepare_dctnet_rgb, _prepare_dkn_rgb, exp6d_loss  # noqa: E402


def strip_module(state):
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and state and all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    return state


class DKNArrayDataset(Dataset):
    """Official DKN array layout, with normalized depth and PIL bicubic LR."""
    def __init__(self, root: Path, train=True):
        self.root = Path(root)
        self.depths = np.load(self.root / ("train_depth_split.npy" if train else "test_depth.npy"), mmap_mode="r")
        self.images = np.load(self.root / ("train_images_split.npy" if train else "test_images_v2.npy"), mmap_mode="r")

    def __len__(self):
        return len(self.depths)

    def __getitem__(self, index):
        depth = np.asarray(self.depths[index], dtype=np.float32)
        rgb = np.asarray(self.images[index], dtype=np.float32).transpose(2, 0, 1) / 255.0
        mn, mx = float(depth.min()), float(depth.max())
        norm = (depth - mn) / max(mx - mn, 1e-6)
        h, w = norm.shape
        lr = np.asarray(Image.fromarray(norm).resize((w // 8, h // 8), Image.Resampling.BICUBIC), dtype=np.float32).copy()
        return (torch.from_numpy(lr[None]), torch.from_numpy(np.ascontiguousarray(rgb)),
                torch.from_numpy(norm[None]), torch.tensor(mn), torch.tensor(mx))


class DCTPatchDataset(Dataset):
    """Legacy exploratory loader retained for compatibility tests only."""
    def __init__(self, root: Path, patches_per_image=16):
        self.root = Path(root)
        self.rgb_dir, self.depth_dir = self.root / "RGB_train", self.root / "depth_train"
        rgb = {p.stem: p for p in self.rgb_dir.glob("*") if p.is_file()}
        depth = {p.stem: p for p in self.depth_dir.glob("*") if p.is_file()}
        self.names = sorted(rgb.keys() & depth.keys(), key=lambda x: int(x) if x.isdigit() else x)
        self.rgb, self.depth = rgb, depth
        self.patches_per_image = int(patches_per_image)

    def __len__(self): return len(self.names) * self.patches_per_image

    def __getitem__(self, index):
        name = self.names[index // self.patches_per_image]
        rgb = np.asarray(Image.open(self.rgb[name]).convert("RGB"), dtype=np.float32) / 255.0
        depth = np.load(self.depth[name]).astype(np.float32)
        if depth.ndim == 3: depth = depth[..., 0]
        h, w = depth.shape
        rng = np.random.default_rng(20260824 + index)
        top, left = int(rng.integers(0, h - 256 + 1)), int(rng.integers(0, w - 256 + 1))
        rgb = rgb[top:top + 256, left:left + 256].transpose(2, 0, 1)
        gt = depth[top:top + 256, left:left + 256]
        mn, mx = float(gt.min()), float(gt.max())
        target = (gt - mn) / max(mx - mn, 1e-6)
        low = np.asarray(Image.fromarray(target).resize((32, 32), Image.Resampling.BICUBIC), dtype=np.float32).copy()
        return torch.from_numpy(low[None]), torch.from_numpy(np.ascontiguousarray(rgb)), torch.from_numpy(target[None]), torch.tensor(mn), torch.tensor(mx)


class DCTOfficialDataset(Dataset):
    """Read DCTNet's official H5 training or processed test layout."""
    def __init__(self, root: Path, patches_per_image=16):
        self.root = Path(root)
        self.h5 = self.root.suffix.lower() in (".h5", ".hdf5")
        if self.h5:
            if h5py is None:
                raise ImportError("h5py is required only for H5 DCTNet data; use an online raw-data directory")
            with h5py.File(self.root, "r") as handle:
                self.names = sorted(handle["HRDepth"].keys(), key=lambda x: int(x))
        else:
            self.rgb_dir, self.depth_dir = self.root / "RGB", self.root / "DepthHR"
            self.lr_dir = self.root / "DepthLrUp"
            self.depth_files = sorted(self.depth_dir.glob("*.npy"))
            self.rgb_files = sorted(self.rgb_dir.glob("*.npy"))
            self.lr_files = sorted(self.lr_dir.glob("*.npy"))
            self.names = [p.stem for p in self.depth_files]
            if not self.names or len(self.rgb_files) != len(self.names) or len(self.lr_files) != len(self.names):
                raise FileNotFoundError(f"Expected DCTNet processed H5 or RGB/DepthHR/DepthLrUp under {self.root}")

    def __len__(self): return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        if self.h5:
            with h5py.File(self.root, "r") as handle:
                low = np.asarray(handle["LRDepth"][name], dtype=np.float32)
                rgb = np.asarray(handle["RGB"][name], dtype=np.float32)
                gt = np.asarray(handle["HRDepth"][name], dtype=np.float32)
        else:
            low = np.asarray(np.load(self.lr_files[index]), dtype=np.float32)
            rgb = np.asarray(np.load(self.rgb_files[index]), dtype=np.float32)
            gt = np.asarray(np.load(self.depth_files[index]), dtype=np.float32)
        if low.ndim == 2: low = low[None]
        if rgb.ndim == 2: rgb = rgb[None]
        if gt.ndim == 2: gt = gt[None]
        mn, mx = float(gt.min()), float(gt.max())
        target = (gt - mn) / max(mx - mn, 1e-6)
        return (torch.from_numpy(np.ascontiguousarray(low)),
                torch.from_numpy(np.ascontiguousarray(rgb)),
                torch.from_numpy(np.ascontiguousarray(target)),
                torch.tensor(mn), torch.tensor(mx))


class DCTOnlineDataset(Dataset):
    """DCTNet preprocessing performed on demand from the raw NYUv2 split."""
    def __init__(self, root: Path, train: bool):
        self.root = Path(root)
        rgb_dir = self.root / ("RGB_train" if train else "nyu_rgb_test")
        depth_dir = self.root / ("depth_train" if train else "nyu_test_gt_16bit")
        rgb = {p.stem: p for p in rgb_dir.iterdir() if p.is_file()}
        depth = {p.stem: p for p in depth_dir.iterdir() if p.is_file()}
        self.items = [(name, rgb[name], depth[name]) for name in
                      sorted(rgb.keys() & depth.keys(), key=lambda x: int(x) if x.isdigit() else x)]
        if not self.items:
            raise FileNotFoundError(f"No RGB/depth pairs found under {rgb_dir} and {depth_dir}")
        self.train = train

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        _, rgb_path, depth_path = self.items[index]
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), np.float32)
        if depth_path.suffix.lower() == ".npy":
            depth = np.load(depth_path).astype(np.float32) * 1000.0
        else:
            depth = np.asarray(Image.open(depth_path), np.float32)
        if self.train:
            rgb = np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize((256, 256), Image.Resampling.BICUBIC), np.float32)
            depth = np.asarray(Image.fromarray(depth).resize((256, 256), Image.Resampling.BICUBIC), np.float32)
        h = min(rgb.shape[0], depth.shape[0]); w = min(rgb.shape[1], depth.shape[1])
        h -= h % 8; w -= w % 8
        rgb = rgb[:h, :w].transpose(2, 0, 1) / 255.0
        depth = depth[:h, :w]
        low = np.asarray(Image.fromarray(depth).resize((w // 8, h // 8), Image.Resampling.BICUBIC), np.float32)
        mn, mx = float(depth.min()), float(depth.max())
        low = (low - mn) / max(mx - mn, 1e-6)
        low_up = np.asarray(Image.fromarray(low).resize((w, h), Image.Resampling.BICUBIC), np.float32)[None]
        # Keep RGB in [0,1] for RF. TrainableOrigin applies DCTNet's
        # ImageNet normalization exactly once at the origin boundary.
        target = (depth - mn) / max(mx - mn, 1e-6)
        return (torch.from_numpy(np.ascontiguousarray(low_up)),
                torch.from_numpy(np.ascontiguousarray(rgb)),
                torch.from_numpy(np.ascontiguousarray(target[None])),
                torch.tensor(mn), torch.tensor(mx))


class TrainableOrigin(nn.Module):
    def __init__(self, method, checkpoint, device):
        super().__init__()
        third = ROOT / "third_party" / ("DKN" if method == "dkn" else "DCTNet")
        sys.path.insert(0, str(third))
        try:
            if method == "dkn":
                from models import DKN
                self.net = DKN(3, 15, residual=True)
            else:
                from model import DCTNet
                self.net = DCTNet()
        finally:
            sys.path.pop(0)
        # Load storage on CPU first; CUDA is initialized only when the complete
        # model is moved below, which avoids checkpoint-load failures on a busy GPU.
        self.net.load_state_dict(strip_module(torch.load(checkpoint, map_location="cpu")), strict=True)
        self.method = method

    def forward(self, depth_lr, rgb):
        depth_hr = _pil_bicubic_upsample(depth_lr, rgb.shape[-2:])
        if self.method == "dkn":
            return self.net((_prepare_dkn_rgb(rgb, bgr=False), depth_hr))
        return self.net(depth_hr, _prepare_dctnet_rgb(rgb))


def load_conv(checkpoint, device):
    model = SurveyDGF(num_features=64)
    model.load_state_dict(strip_module(torch.load(checkpoint, map_location="cpu")), strict=True)
    return model


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def build_official_scheduler(method, optimizer):
    """Match the scheduler settings from each upstream training script."""
    if method == "dkn":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.2)
    if method == "dctnet":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    return None


def origin_task_loss(method, coarse, target):
    """Original supervised objective used by each origin implementation."""
    if method in ("conv", "dkn"):
        return F.l1_loss(coarse, target)
    if method == "dctnet":
        return F.mse_loss(coarse, target)
    raise ValueError(f"Unsupported origin loss method: {method}")


def macro_metrics(model, origin, loader, device, crop=6, origin_grad=False):
    model.eval(); origin.eval(); rows = []
    with torch.inference_mode():
        for batch in loader:
            depth, rgb, target, mn, mx = batch[:5]
            depth, rgb, target = depth.to(device), rgb.to(device), target.to(device)
            if origin_grad:
                coarse = origin(depth, rgb)
            else:
                with torch.no_grad():
                    coarse = origin(depth, rgb)
            # Match Exp1's ConvGuided->RF protocol: RF always receives the
            # normalized coarse prediction clipped to the valid depth range.
            coarse = coarse.clamp(0.0, 1.0)
            pred = model(rgb, coarse)["final"].clamp(0, 1)
            scale = (mx.to(device) - mn.to(device)).view(-1, 1, 1, 1)
            diff_c, diff_p = (coarse - target) * scale, (pred - target) * scale
            if crop:
                diff_c, diff_p = diff_c[..., crop:-crop, crop:-crop], diff_p[..., crop:-crop, crop:-crop]
            for i in range(depth.shape[0]):
                rows.append({"coarse_rmse_mm": float(diff_c[i].square().mean().sqrt()), "rf_rmse_mm": float(diff_p[i].square().mean().sqrt()),
                             "coarse_mae_mm": float(diff_c[i].abs().mean()), "rf_mae_mm": float(diff_p[i].abs().mean())})
    return rows


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.method == "conv":
        train = NYUv2FullImageDataset(args.data_root, "train", 8, indices=range(1000), augment=True, depth_unit_scale=1000.0)
        test = NYUv2FullImageDataset(args.test_root, "test", 8, depth_unit_scale=1.0)
        origin = load_conv(args.origin_checkpoint, device)
    elif args.method == "dkn":
        # The 1000 training pairs are the complete DKN training pool. The
        # independent 449-pair test set is never used for model selection.
        train = DKNArrayDataset(args.dkn_data, True); test = DKNArrayDataset(args.dkn_data, False)
        origin = TrainableOrigin("dkn", args.origin_checkpoint, device)
    else:
        if args.dct_train is None or args.dct_test is None:
            raise ValueError("DCTNet requires --dct-train and --dct-test official processed data paths")
        # Read raw NYUv2 files and perform DCTNet preprocessing per sample.
        train = DCTOnlineDataset(args.dct_train, train=True)
        test = DCTOnlineDataset(args.dct_test, train=False)
        origin = TrainableOrigin("dctnet", args.origin_checkpoint, device)
    rf = Exp6DRefine3Stage()
    origin, rf = origin.to(device), rf.to(device)
    train_loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, 1, shuffle=False, num_workers=0)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and args.batch_size >= torch.cuda.device_count():
        origin, rf = nn.DataParallel(origin), nn.DataParallel(rf)
    if args.freeze_origin:
        origin.eval()
        for parameter in origin.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(rf.parameters(), lr=args.rf_lr)
    else:
        optimizer = torch.optim.Adam([{"params": origin.parameters(), "lr": args.origin_lr}, {"params": rf.parameters(), "lr": args.rf_lr}])
    scheduler = build_official_scheduler(args.method, optimizer)
    output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best = float("inf")
    if args.resume is not None:
        resume_state = torch.load(args.resume, map_location=device)
        unwrap(origin).load_state_dict(resume_state["origin"])
        unwrap(rf).load_state_dict(resume_state["rf"])
        start_epoch = int(resume_state.get("epoch", 0))
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        if scheduler is not None and "lr_schedule" in resume_state:
            scheduler.load_state_dict(resume_state["lr_schedule"])
        print(json.dumps({"resume": str(args.resume), "start_epoch": start_epoch, "target_epochs": args.epochs}))
    for epoch in range(start_epoch, args.epochs):
        if not args.freeze_origin:
            origin.train()
        else:
            origin.eval()
        rf.train()
        for depth, rgb, target, *_ in tqdm(train_loader, desc=f"exp3-{args.method} {epoch+1}/{args.epochs}"):
            depth, rgb, target = depth.to(device), rgb.to(device), target.to(device)
            if args.freeze_origin:
                with torch.no_grad():
                    coarse = origin(depth, rgb)
            else:
                coarse = origin(depth, rgb)
            coarse = coarse.clamp(0.0, 1.0)
            out = rf(rgb, coarse)
            loss = exp6d_loss(out, target, coarse)
            if not args.freeze_origin:
                # Exp7C fidelity term: keep the jointly updated origin useful
                # instead of letting it drift while RF learns the residual.
                loss = (loss
                        + 0.5 * F.smooth_l1_loss(coarse, target)
                        + args.origin_loss_weight * origin_task_loss(args.method, coarse, target))
            optimizer.zero_grad(); loss.backward()
            if args.method == "dctnet":
                nn.utils.clip_grad_norm_(list(origin.parameters()) + list(rf.parameters()), max_norm=1.0, norm_type=2)
            if args.method == "dkn":
                # Official DKN calls StepLR once per batch before optimizer.step().
                scheduler.step()
            optimizer.step()
        if args.method == "dctnet":
            # Official DCTNet calls StepLR once after each epoch.
            scheduler.step()
        # The project intentionally uses the 449-image benchmark as the
        # selection set for this score-optimization experiment.
        rows = macro_metrics(rf, origin, test_loader, device, args.crop, origin_grad=False)
        test_rmse = float(np.mean([r["rf_rmse_mm"] for r in rows]))
        state = {
            "origin": unwrap(origin).state_dict(),
            "rf": unwrap(rf).state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_schedule": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch + 1,
            "freeze_origin": args.freeze_origin,
            "test_rf_rmse_mm": test_rmse,
        }
        torch.save(state, output / "exp3_joint_last.pth")
        if test_rmse < best:
            best = test_rmse
            state["best_test_rf_rmse_mm"] = best
            torch.save(state, output / "exp3_joint_best.pth")
        print(json.dumps({"epoch": epoch + 1, "train_loss": float(loss.item()), "test_rf_rmse_mm": test_rmse, "best_test_rf_rmse_mm": best}))
    # Always report the test-selected checkpoint after training.  The `last`
    # checkpoint is retained for resume/debugging, but must not replace the
    # best checkpoint in the final metrics.
    state_path = output / "exp3_joint_best.pth"
    if not state_path.exists():
        state_path = output / "exp3_joint_last.pth"
    state = torch.load(state_path, map_location=device)
    # Checkpoints are saved from the unwrapped modules, so restore into the
    # unwrapped modules even when training used DataParallel.
    unwrap(origin).load_state_dict(state["origin"])
    unwrap(rf).load_state_dict(state["rf"])
    rows = macro_metrics(rf, origin, test_loader, device, args.crop, origin_grad=False)
    summary = {"method": args.method, "epochs": args.epochs, "images": len(rows), "coarse_rmse_cm": float(np.mean([r["coarse_rmse_mm"] for r in rows]) / 10),
               "rf_rmse_cm": float(np.mean([r["rf_rmse_mm"] for r in rows]) / 10), "coarse_mae_cm": float(np.mean([r["coarse_mae_mm"] for r in rows]) / 10),
               "rf_mae_cm": float(np.mean([r["rf_mae_mm"] for r in rows]) / 10)}
    (output / "test_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("conv", "dkn", "dctnet"), required=True)
    p.add_argument("--origin-checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path(r"D:\nuoqi\halo-free\data_NYUv2\NYUv2-resplit\NYUv2"))
    p.add_argument("--test-root", type=Path, default=Path(r"D:\nuoqi\halo-free\data_NYUv2\benchmark_test_449"))
    p.add_argument("--dkn-data", type=Path, default=ROOT / "exp1/data/dkn_nyuv2_1000_449")
    p.add_argument("--dct-train", type=Path, default=None, help="official DCTNet processed training data")
    p.add_argument("--dct-test", type=Path, default=None, help="official DCTNet processed test data")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--freeze-origin", action="store_true", help="freeze origin and train RF only")
    p.add_argument("--epochs", type=int, default=80); p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--origin-lr", type=float, default=3e-4); p.add_argument("--rf-lr", type=float, default=3e-3)
    p.add_argument("--origin-loss-weight", type=float, default=1.0)
    p.add_argument("--crop", type=int, default=6); p.add_argument("--device", default="cuda:0")
    main(p.parse_args())
