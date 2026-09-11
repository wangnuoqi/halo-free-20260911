"""Train the three-stage residual refinement network on full-image NYUv2."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from nyuv2_full_dataset import NYUv2FullImageDataset
from convguided_model import SurveyDGF


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(r"D:\nuoqi\halo-free\data_NYUv2\NYUv2-resplit\NYUv2")
DEFAULT_TEST_ROOT = Path(r"D:\nuoqi\halo-free\data_NYUv2\benchmark_test_449")
DEFAULT_CONV_BASE_CHECKPOINT = PROJECT_ROOT / "models" / "conv_survey_dgf_x4" / "rgb_guided_depth_x4_best.pth"
DEFAULT_OUTPUT_DIR = Path(r"D:\nuoqi\halo-free\Decoupled_Learning\20260623_RGBDsr\models\rf_3stage_nyuv2_x4")


class Stage1DeltaNet(nn.Module):
    def __init__(self, in_channels=5, base_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, base_channels, 3, padding=1, bias=False), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1, bias=False),
        )

    def forward(self, x_raw, x_coarse):
        g, _, _ = compute_sobel_grad(x_coarse, normalize=True)
        return self.net(torch.cat([x_raw, x_coarse, g], dim=1))


class ABAdjustNet(nn.Module):
    def __init__(self, base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True):
        super().__init__()
        self.a_scale, self.b_scale = a_scale, b_scale
        self.use_delta_as_input = use_delta_as_input
        in_channels = 6 if use_delta_as_input else 5  # RGB + coarse + gradient + delta
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, base_channels, 3, padding=1, bias=False), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 2, 1, bias=True),
        )

    def forward(self, x_raw, x_coarse_k, delta_prev=None):
        g, _, _ = compute_sobel_grad(x_coarse_k, normalize=True)
        parts = [x_raw, x_coarse_k, g]
        if self.use_delta_as_input:
            if delta_prev is None:
                raise ValueError("delta_prev is required when use_delta_as_input=True")
            parts.append(delta_prev)
        ab = self.net(torch.cat(parts, dim=1))
        return 1.0 + self.a_scale * torch.tanh(ab[:, 0:1]), self.b_scale * torch.tanh(ab[:, 1:2])


class CascadeRefine3Stage(nn.Module):
    def __init__(self, base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True):
        super().__init__()
        self.stage1 = Stage1DeltaNet(base_channels=base_channels)
        self.stage2 = ABAdjustNet(base_channels, a_scale, b_scale, use_delta_as_input)
        self.stage3 = ABAdjustNet(base_channels, a_scale, b_scale, use_delta_as_input)

    def forward(self, x_raw, x_coarse0):
        delta1 = self.stage1(x_raw, x_coarse0)
        coarse1 = x_coarse0 + delta1
        a2, b2 = self.stage2(x_raw, coarse1, delta1)
        delta2 = a2 * delta1 + b2
        coarse2 = x_coarse0 + delta2
        a3, b3 = self.stage3(x_raw, coarse2, delta2)
        delta3 = a3 * delta2 + b3
        return {"delta1": delta1, "coarse1": coarse1, "delta2": delta2, "coarse2": coarse2,
                "delta3": delta3, "final": x_coarse0 + delta3, "a2": a2, "b2": b2, "a3": a3, "b3": b3}


def compute_sobel_grad(x, eps=1e-6, normalize=False):
    kx = x.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
    gx, gy = F.conv2d(x, kx, padding=1), F.conv2d(x, ky, padding=1)
    g = torch.sqrt(gx.square() + gy.square() + eps)
    if normalize:
        g = g / (g.amax(dim=(2, 3), keepdim=True) + eps)
    return g, gx, gy


def edge_guided_loss(pred, target):
    g, _, _ = compute_sobel_grad(target, normalize=True)
    threshold = g.flatten(1).quantile(0.90, dim=1).view(-1, 1, 1, 1)
    edge = (g > threshold).float().detach()
    smooth = 1.0 - edge
    pixel = (pred - target).abs()
    edge_loss = (pixel * edge).sum() / edge.sum().clamp_min(1.0)
    smooth_loss = (pixel * smooth).sum() / smooth.sum().clamp_min(1.0)
    _, pgx, pgy = compute_sobel_grad(pred, normalize=True)
    _, tgx, tgy = compute_sobel_grad(target, normalize=True)
    grad = ((pgx - tgx).abs() + (pgy - tgy).abs()) * edge
    grad_loss = grad.sum() / edge.sum().clamp_min(1.0)
    return edge_loss + smooth_loss + grad_loss


def restore_depth(x, depth_min, depth_max):
    return x * (depth_max - depth_min).view(-1, 1, 1, 1) + depth_min.view(-1, 1, 1, 1)


@torch.inference_mode()
def predict_conv_base(model, depth_lr, rgb):
    """Generate the frozen conv-base HR coarse depth consumed by the refine net."""
    return model(depth_lr, rgb).clamp(0.0, 1.0)


def load_conv_base(checkpoint, device, num_features=64):
    model = SurveyDGF(num_features=num_features).to(device)
    state = torch.load(checkpoint, map_location=device)
    state = state.get("model_state_dict", state) if isinstance(state, dict) else state
    try:
        model.load_state_dict(state)
    except RuntimeError:
        if all(key.startswith("module.") for key in state):
            model.load_state_dict({key[7:]: value for key, value in state.items()})
        else:
            raise
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def evaluate(model, loader, device, conv_base, crop=6):
    model.eval(); sq = ab = pixels = 0.0
    with torch.inference_mode():
        for depth_lr, rgb, target, depth_min, depth_max, _ in loader:
            rgb, target = rgb.to(device), target.to(device)
            coarse = predict_conv_base(conv_base, depth_lr.to(device), rgb)
            pred = model(rgb, coarse)["final"].clamp(0, 1)
            pred, target = restore_depth(pred, depth_min.to(device), depth_max.to(device)), restore_depth(target, depth_min.to(device), depth_max.to(device))
            if crop: pred, target = pred[..., crop:-crop, crop:-crop], target[..., crop:-crop, crop:-crop]
            diff = pred - target; sq += diff.square().sum().item(); ab += diff.abs().sum().item(); pixels += diff.numel()
    return {"rmse_mm": float(np.sqrt(sq / pixels)), "rmse_cm": float(np.sqrt(sq / pixels) / 10), "mae_mm": float(ab / pixels), "mae_cm": float(ab / pixels / 10), "pixels": int(pixels)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    p.add_argument("--scale", type=int, choices=(4, 8, 16), default=4)
    p.add_argument("--epochs", type=int, default=80); p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--device", default="cuda:0")
    p.add_argument("--conv-base-checkpoint", type=Path, default=DEFAULT_CONV_BASE_CHECKPOINT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def resolve_device(requested):
    """Match the project training scripts: CUDA requests fall back cleanly to CPU."""
    try:
        cuda_ok = torch.cuda.is_available()
    except Exception as exc:
        print(f"Warning: CUDA initialization failed ({exc}); using CPU.")
        cuda_ok = False
    if requested.startswith("cuda") and cuda_ok:
        return torch.device(requested)
    if requested.startswith("cuda") and not cuda_ok:
        print("Warning: CUDA is unavailable; falling back to CPU.")
    return torch.device("cpu")


def main(args):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = resolve_device(args.device)
    train_ds = NYUv2FullImageDataset(args.data_root, "train", args.scale, indices=range(900), augment=True, depth_unit_scale=1000.0)
    val_ds = NYUv2FullImageDataset(args.data_root, "train", args.scale, indices=range(900, 1000), depth_unit_scale=1000.0)
    test_ds = NYUv2FullImageDataset(args.test_root, "test", args.scale, depth_unit_scale=1.0)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, 1, shuffle=False, num_workers=0); test_loader = DataLoader(test_ds, 1, shuffle=False, num_workers=0)
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    model = CascadeRefine3Stage()
    conv_base = load_conv_base(args.conv_base_checkpoint, device)
    if gpu_count > 1 and args.batch_size >= gpu_count:
        model = torch.nn.DataParallel(model)
        conv_base = torch.nn.DataParallel(conv_base)
    model = model.to(device); conv_base = conv_base.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True); best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        for depth_lr, rgb, target, *_ in tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            rgb, target = rgb.to(device), target.to(device)
            coarse = predict_conv_base(conv_base, depth_lr.to(device), rgb)
            out = model(rgb, coarse); gt_delta = target - coarse
            loss = sum(edge_guided_loss(out[f"delta{i}"], gt_delta) for i in (1, 2, 3)) / 3
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        metrics = evaluate(model, val_loader, device, conv_base=conv_base)
        print(json.dumps({"epoch": epoch + 1, **metrics}, sort_keys=True))
        if metrics["rmse_cm"] < best:
            best = metrics["rmse_cm"]; torch.save(model.state_dict(), args.output_dir / "rf_3stage_best.pth")
    test_metrics = evaluate(model, test_loader, device, conv_base=conv_base)
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n")
    print(json.dumps({"final_test": test_metrics}, indent=2))


if __name__ == "__main__":
    main(parse_args())
