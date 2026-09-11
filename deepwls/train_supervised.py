import argparse
import datetime
import os
import random
from pathlib import Path

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[override]
        def __init__(self, iterable=None, total=None, desc=None, dynamic_ncols=None, leave=None):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            pass

from loss_function.wls_aligned_supervised_loss import WLSAlignedSupervisedLoss
from net.fcnBn import FcnBn
from util.dataset import PairedImageDataset

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        class SummaryWriter:  # type: ignore[override]
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def add_images(self, *args, **kwargs):
                pass

            def close(self):
                pass


DEFAULT_DATA_ROOT = Path(r"D:\nuoqi\S2R-HDR\data\mixed_lavel_singlehdr")
DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "checkpoints_supervised"
DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "runs_supervised"
DEFAULT_WLS_LAMBDA = 0.25
DEFAULT_WLS_ALPHA = 1.2
DEFAULT_SMOOTH_WEIGHT = 1.0

if os.name != "nt":
    DEFAULT_DATA_ROOT = Path("/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr")


def parse_args():
    def parse_patch_size(value):
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"none", "full", "0"}:
            return None
        size = int(text)
        if size <= 0:
            return None
        return size

    parser = argparse.ArgumentParser(
        description="Supervised deepwls training against fixed-parameter WLS labels."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16, help="Global batch size for DataParallel.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--patch-size",
        type=parse_patch_size,
        default=None,
        help="Training crop size. Use 0/none/full to train on full-resolution images.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wls-lambda", type=float, default=DEFAULT_WLS_LAMBDA)
    parser.add_argument("--wls-alpha", type=float, default=DEFAULT_WLS_ALPHA)
    parser.add_argument("--smooth-weight", type=float, default=DEFAULT_SMOOTH_WEIGHT)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume-path", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(args):
    train_dataset = PairedImageDataset(
        args.data_root / "traininput",
        args.data_root / "trainlabel",
        split="train",
        patch_size=args.patch_size,
    )
    test_dataset = PairedImageDataset(
        args.data_root / "testinput",
        args.data_root / "testlabel",
        split="test",
        patch_size=None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader


def ensure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("No CUDA devices are visible. Please set CUDA_VISIBLE_DEVICES correctly.")
    return torch.device("cuda"), list(range(device_count))


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def evaluate(model, loader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_recon = 0.0
    total_smooth = 0.0

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss, recon_loss, smooth_loss = criterion(
                inputs,
                outputs,
                targets,
                args.wls_lambda,
                args.wls_alpha,
                smooth_weight=args.smooth_weight,
            )
            mse = torch.mean((outputs - targets) ** 2)
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_smooth += smooth_loss.item()
            total_mse += mse.item()

    num_batches = max(len(loader), 1)
    return (
        total_loss / num_batches,
        total_mse / num_batches,
        total_recon / num_batches,
        total_smooth / num_batches,
    )


def save_checkpoint(save_dir, epoch, model, optimizer, best_metric, is_best):
    save_dir.mkdir(parents=True, exist_ok=True)
    full_path = save_dir / "deepwls_supervised_full.pth"
    last_path = save_dir / "deepwls_supervised_last.pth"
    best_path = save_dir / "deepwls_supervised_best.pth"

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
        },
        full_path,
    )
    torch.save(unwrap_model(model).state_dict(), last_path)
    if is_best:
        torch.save(unwrap_model(model).state_dict(), best_path)


def maybe_resume(args, model, optimizer, device):
    start_epoch = 0
    best_metric = float("inf")
    resume_path = args.resume_path or (args.save_dir / "deepwls_supervised_full.pth")

    if args.no_resume or not resume_path.exists():
        print("=> No checkpoint found, training from scratch.")
        return start_epoch, best_metric

    print(f"=> Resuming training from checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    best_metric = checkpoint.get("best_metric", float("inf"))
    print(f"=> Resumed from epoch {start_epoch}, best_metric = {best_metric:.6f}")
    return start_epoch, best_metric


def main():
    args = parse_args()
    set_seed(args.seed)
    device, device_ids = ensure_cuda()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"=> Supervised fixed-WLS config: wls_lambda={args.wls_lambda}, "
        f"wls_alpha={args.wls_alpha}, smooth_weight={args.smooth_weight}, "
        f"patch_size={args.patch_size or 'full'}"
    )
    train_loader, test_loader = build_dataloaders(args)

    model = FcnBn()
    model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0]).to(device)
    print(f"=> Using DataParallel device_ids={device_ids}")

    criterion = WLSAlignedSupervisedLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch, best_metric = maybe_resume(args, model, optimizer, device)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(args.log_dir / f"deepwls_supervised_{timestamp}"))

    epoch_bar = tqdm(range(start_epoch, args.epochs), desc="epochs", dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0

        batch_bar = tqdm(
            train_loader,
            desc=f"train {epoch + 1}/{args.epochs}",
            dynamic_ncols=True,
            leave=False,
        )
        running_recon = 0.0
        running_smooth = 0.0
        for batch_idx, (inputs, targets, _) in enumerate(batch_bar, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss, recon_loss, smooth_loss = criterion(
                inputs,
                outputs,
                targets,
                args.wls_lambda,
                args.wls_alpha,
                smooth_weight=args.smooth_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_smooth += smooth_loss.item()
            batch_bar.set_postfix(
                loss=f"{running_loss / batch_idx:.6f}",
                recon=f"{running_recon / batch_idx:.6f}",
                smooth=f"{running_smooth / batch_idx:.6f}",
            )

        avg_train_loss = running_loss / max(len(train_loader), 1)
        avg_train_recon = running_recon / max(len(train_loader), 1)
        avg_train_smooth = running_smooth / max(len(train_loader), 1)
        avg_val_loss, avg_val_mse, avg_val_recon, avg_val_smooth = evaluate(model, test_loader, criterion, device, args)
        is_best = avg_val_mse < best_metric
        if is_best:
            best_metric = avg_val_mse

        save_checkpoint(args.save_dir, epoch, model, optimizer, best_metric, is_best)
        epoch_bar.set_postfix(
            train_loss=f"{avg_train_loss:.6f}",
            train_recon=f"{avg_train_recon:.6f}",
            val_mse=f"{avg_val_mse:.6f}",
            best=f"{best_metric:.6f}",
        )

        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={avg_train_loss:.6f} "
            f"train_recon={avg_train_recon:.6f} "
            f"train_smooth={avg_train_smooth:.6f} "
            f"val_loss={avg_val_loss:.6f} "
            f"val_recon={avg_val_recon:.6f} "
            f"val_smooth={avg_val_smooth:.6f} "
            f"val_mse={avg_val_mse:.6f} "
            f"best_val_mse={best_metric:.6f}"
        )

        writer.add_scalar("train/loss", avg_train_loss, epoch + 1)
        writer.add_scalar("train/recon", avg_train_recon, epoch + 1)
        writer.add_scalar("train/smooth", avg_train_smooth, epoch + 1)
        writer.add_scalar("val/loss", avg_val_loss, epoch + 1)
        writer.add_scalar("val/recon", avg_val_recon, epoch + 1)
        writer.add_scalar("val/smooth", avg_val_smooth, epoch + 1)
        writer.add_scalar("val/mse", avg_val_mse, epoch + 1)
        writer.add_images("train/input", inputs[:4], epoch + 1)
        writer.add_images("train/target", targets[:4], epoch + 1)
        writer.add_images("train/output", outputs[:4], epoch + 1)

    writer.close()


if __name__ == "__main__":
    main()
