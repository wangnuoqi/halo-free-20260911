"""Evaluate the frozen conv-base followed by the trained three-stage RF model."""

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
from test_convguided_base import compute_image_metrics, restore_depth
from train_rf_3stagev3 import CascadeRefine3Stage, predict_conv_base


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent.parent / "data_NYUv2" / "benchmark_test_449"
DEFAULT_CONV_CHECKPOINT = PROJECT_ROOT / "models" / "v3_survey_dgf_fullimage_per_image_norm_x4" / "rgb_guided_depth_x4_best.pth"
DEFAULT_RF_CHECKPOINT = PROJECT_ROOT / "models" / "rf_3stage_nyuv2_x4" / "rf_3stage_best.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test_results" / "rf_3stage_nyuv2_x4"


def _load_state(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    candidates = [state]
    if state and all(key.startswith("module.") for key in state):
        candidates.append({key[7:]: value for key, value in state.items()})
    last_error = None
    for candidate in candidates:
        try:
            model.load_state_dict(candidate)
            return
        except RuntimeError as error:
            last_error = error
    raise RuntimeError(f"Could not load checkpoint {checkpoint_path}: {last_error}")


def _resolve_device(requested: str | None) -> torch.device:
    if requested and requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if requested and requested.startswith("cuda"):
        print("Warning: CUDA is unavailable; evaluating on CPU.")
    return torch.device("cpu")


def build_summary(rf_checkpoint: Path, data_root: Path, scale: int, crop: int, rows: list[dict]) -> dict:
    pixels = sum(row["pixels"] for row in rows)
    if not pixels:
        raise RuntimeError("No pixels were evaluated")

    def aggregate(prefix: str, key: str) -> float:
        # Benchmark protocol: compute each image metric first, then macro-average
        # across images. Do not merge all pixels or weight images by resolution.
        return float(np.mean([row[f"{prefix}_{key}_mm"] for row in rows]))

    coarse_rmse_mm = aggregate("coarse", "rmse")
    coarse_mae_mm = aggregate("coarse", "mae")
    rf_rmse_mm = aggregate("rf", "rmse")
    rf_mae_mm = aggregate("rf", "mae")
    return {
        "checkpoint": str(Path(rf_checkpoint).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "scale": scale,
        "crop": crop,
        "images": len(rows),
        "pixels": pixels,
        "coarse_rmse_mm": coarse_rmse_mm,
        "coarse_rmse_cm": coarse_rmse_mm / 10.0,
        "coarse_mae_mm": coarse_mae_mm,
        "coarse_mae_cm": coarse_mae_mm / 10.0,
        "rf_rmse_mm": rf_rmse_mm,
        "rf_rmse_cm": rf_rmse_mm / 10.0,
        "rf_mae_mm": rf_mae_mm,
        "rf_mae_cm": rf_mae_mm / 10.0,
        "rmse_improvement_cm": (coarse_rmse_mm - rf_rmse_mm) / 10.0,
        "mae_improvement_cm": (coarse_mae_mm - rf_mae_mm) / 10.0,
        "unit": "millimeters and centimeters",
    }


def save_results(output_dir: Path, summary: dict, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    fields = ["name", "coarse_rmse_mm", "coarse_rmse_cm", "rf_rmse_mm", "rf_rmse_cm", "coarse_mae_mm", "coarse_mae_cm", "rf_mae_mm", "rf_mae_cm", "pixels"]
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def evaluate(args: argparse.Namespace) -> dict:
    device = _resolve_device(args.device)
    conv_base = SurveyDGF(num_features=args.num_features).to(device)
    _load_state(conv_base, args.conv_checkpoint, device)
    conv_base.eval()
    rf_model = CascadeRefine3Stage(base_channels=args.base_channels).to(device)
    _load_state(rf_model, args.rf_checkpoint, device)
    rf_model.eval()

    dataset = NYUv2FullImageDataset(args.data_root, split="test", scale=args.scale, depth_unit_scale=1.0, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    rows = []
    with torch.inference_mode():
        for depth_lr, rgb, target, depth_min, depth_max, names in loader:
            depth_lr, rgb, target = depth_lr.to(device), rgb.to(device), target.to(device)
            coarse = predict_conv_base(conv_base, depth_lr, rgb)
            refined = rf_model(rgb, coarse)["final"].clamp(0.0, 1.0)
            coarse_raw = restore_depth(coarse, depth_min, depth_max)
            refined_raw = restore_depth(refined, depth_min, depth_max)
            target_raw = restore_depth(target, depth_min, depth_max)
            for index, name in enumerate(names):
                coarse_metrics = compute_image_metrics(coarse_raw[index:index + 1], target_raw[index:index + 1], args.crop)
                rf_metrics = compute_image_metrics(refined_raw[index:index + 1], target_raw[index:index + 1], args.crop)
                rows.append({
                    "name": str(name),
                    "coarse_rmse_mm": coarse_metrics["rmse"], "coarse_rmse_cm": coarse_metrics["rmse"] / 10.0,
                    "rf_rmse_mm": rf_metrics["rmse"], "rf_rmse_cm": rf_metrics["rmse"] / 10.0,
                    "coarse_mae_mm": coarse_metrics["mae"], "coarse_mae_cm": coarse_metrics["mae"] / 10.0,
                    "rf_mae_mm": rf_metrics["mae"], "rf_mae_cm": rf_metrics["mae"] / 10.0,
                    "pixels": coarse_metrics["pixels"],
                })
    summary = build_summary(args.rf_checkpoint, args.data_root, args.scale, args.crop, rows)
    save_results(args.output_dir, summary, rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--conv-checkpoint", type=Path, default=DEFAULT_CONV_CHECKPOINT)
    parser.add_argument("--rf-checkpoint", type=Path, default=DEFAULT_RF_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scale", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--crop", type=int, default=6)
    parser.add_argument("--num-features", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))
