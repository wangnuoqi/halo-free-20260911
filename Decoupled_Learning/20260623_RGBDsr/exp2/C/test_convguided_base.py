"""Evaluate a trained SurveyDGF model on the full-image NYUv2 protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from convguided_model import SurveyDGF
from nyuv2_full_dataset import NYUv2FullImageDataset


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "exp2_C_detail_gate_x4" / "rgb_guided_depth_x4_best.pth"
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent.parent / "data_NYUv2" / "benchmark_test_449"


def restore_depth(normalized: torch.Tensor, depth_min: torch.Tensor, depth_max: torch.Tensor) -> torch.Tensor:
    """Undo the per-image min/max normalization used by NYUv2FullImageDataset."""
    depth_min = depth_min.to(device=normalized.device, dtype=normalized.dtype).view(-1, 1, 1, 1)
    depth_max = depth_max.to(device=normalized.device, dtype=normalized.dtype)
    depth_range = (depth_max - depth_min.view(-1)).to(device=normalized.device, dtype=normalized.dtype)
    return normalized * depth_range.view(-1, 1, 1, 1) + depth_min


def compute_image_metrics(prediction: torch.Tensor, target: torch.Tensor, crop: int = 6) -> dict:
    """Return physical-unit RMSE/MAE for one or more BCHW images."""
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} vs {target.shape}")
    if crop < 0:
        raise ValueError("crop must be non-negative")
    if crop:
        if prediction.shape[-2] <= 2 * crop or prediction.shape[-1] <= 2 * crop:
            raise ValueError(f"crop={crop} is too large for image shape {prediction.shape[-2:]}")
        prediction = prediction[..., crop:-crop, crop:-crop]
        target = target[..., crop:-crop, crop:-crop]
    diff = prediction - target
    return {
        "rmse": float(torch.sqrt(torch.mean(diff.square())).item()),
        "mae": float(torch.mean(diff.abs()).item()),
        "pixels": int(diff.numel()),
    }


def save_metrics(output_dir: Path, summary: dict, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    fieldnames = ["name", "rmse_mm", "rmse_cm", "mae_mm", "mae_cm", "pixels"]
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)


def build_summary(checkpoint: Path, data_root: Path, split: str, scale: int, crop: int, rows: list[dict]) -> dict:
    pixel_count = sum(row["pixels"] for row in rows)
    if not pixel_count:
        raise RuntimeError("No pixels were evaluated")
    # Official protocol: compute metrics per image, then macro-average images.
    rmse_mm = float(np.mean([row["rmse_mm"] for row in rows]))
    mae_mm = float(np.mean([row["mae_mm"] for row in rows]))
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "split": split,
        "scale": scale,
        "images": len(rows),
        "crop": crop,
        "rmse_mm": rmse_mm,
        "rmse_cm": rmse_mm / 10.0,
        "mae_mm": mae_mm,
        "mae_cm": mae_mm / 10.0,
        "pixels": pixel_count,
        "unit": "millimeters and centimeters",
    }


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Training may have used DataParallel while inference uses one GPU/CPU.
        if all(key.startswith("module.") for key in state_dict):
            model.load_state_dict({key[7:]: value for key, value in state_dict.items()})
        else:
            raise


def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SurveyDGF(num_features=args.num_features).to(device)
    _load_checkpoint(model, Path(args.checkpoint), device)
    model.eval()

    dataset = NYUv2FullImageDataset(
        args.data_root,
        split=args.split,
        scale=args.scale,
        depth_unit_scale=args.depth_unit_scale,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    rows = []
    squared_error = 0.0
    absolute_error = 0.0
    pixel_count = 0
    with torch.inference_mode():
        for depth_lr, rgb_hr, depth_label, depth_min, depth_max, names in loader:
            prediction = model(depth_lr.to(device), rgb_hr.to(device)).clamp(0.0, 1.0)
            prediction_raw = restore_depth(prediction, depth_min, depth_max)
            target_raw = restore_depth(depth_label.to(device), depth_min, depth_max)
            for index, name in enumerate(names):
                item = compute_image_metrics(prediction_raw[index:index + 1], target_raw[index:index + 1], args.crop)
                rows.append({
                    "name": str(name),
                    "rmse_mm": item["rmse"],
                    "rmse_cm": item["rmse"] / 10.0,
                    "mae_mm": item["mae"],
                    "mae_cm": item["mae"] / 10.0,
                    "pixels": item["pixels"],
                })
                pixel_count += item["pixels"]

    summary = build_summary(args.checkpoint, args.data_root, args.split, args.scale, args.crop, rows)
    save_metrics(Path(args.output_dir), summary, rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scale", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--depth-unit-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--crop", type=int, default=6)
    parser.add_argument("--num-features", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "test_results" / "exp2_C_detail_gate_x4")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.crop < 0:
        parser.error("batch-size must be positive; workers and crop must be non-negative")
    return args


if __name__ == "__main__":
    result = evaluate(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
