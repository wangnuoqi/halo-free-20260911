"""Generate depth-aware RGB-D super-resolution visualizations.

The numeric outputs are restored to millimeters.  Color maps use each image's
GT min/max so models are compared on the same physical depth range; error maps
use one configurable fixed range across all models and images.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parents[1] / "20260623_RGBDsr"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(PROJECT / "exp1") not in sys.path:
    sys.path.insert(0, str(PROJECT / "exp1"))

from exp1.convguided_model import SurveyDGF  # noqa: E402
from exp1.nyuv2_full_dataset import NYUv2FullImageDataset  # noqa: E402
from exp1.train_rf_3stagev3 import CascadeRefine3Stage  # noqa: E402
from exp3.train_exp3_joint import strip_module  # noqa: E402
from thirdparty_refine_models import ThirdPartyCoarse  # noqa: E402


MODELS = HERE / "models"
DEFAULT_DATA = HERE.parents[2] / "data_NYUv2" / "benchmark_test_449"


def load_state(model, path: Path, key: str | None = None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if key:
        state = state[key]
    state = strip_module(state)
    model.load_state_dict(state, strict=True)
    return model


def make_models(device: torch.device):
    """Load only copied checkpoints from this visualization directory."""
    models = {}

    conv = load_state(SurveyDGF(num_features=64), MODELS / "conv_x8_exp1_best.pth").to(device).eval()
    conv_rf = load_state(CascadeRefine3Stage(), MODELS / "rf_original_conv_x8_exp1_best.pth").to(device).eval()
    models["conv_rf"] = ("rf", conv, conv_rf)

    dkn = ThirdPartyCoarse("dkn", MODELS / "dkn_official_x8.pth", device,
                           clamp_output=True, bgr_input=False).eval()
    dct = ThirdPartyCoarse("dctnet", MODELS / "dctnet_official_x8.pth", device,
                           clamp_output=True).eval()
    models["dkn"] = ("coarse_thirdparty", dkn, None)
    models["dctnet"] = ("coarse_thirdparty", dct, None)

    return models


def colorize(depth_mm: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = np.clip((depth_mm - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap(np.uint8(np.round(scaled * 255)), cv2.COLORMAP_TURBO)


def label_tile(tile: np.ndarray, label: str) -> np.ndarray:
    out = tile.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def resize_tile(tile: np.ndarray, width: int = 320) -> np.ndarray:
    height = max(1, round(tile.shape[0] * width / tile.shape[1]))
    return cv2.resize(tile, (width, height), interpolation=cv2.INTER_AREA)


def image_metrics(pred: np.ndarray, target: np.ndarray, crop: int):
    if crop:
        pred = pred[crop:-crop, crop:-crop]
        target = target[crop:-crop, crop:-crop]
    diff = pred - target
    return float(np.sqrt(np.mean(diff * diff))), float(np.mean(np.abs(diff)))


def edge_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, crop: int) -> float:
    """Compute edge-only MAE after applying the same crop to both arrays."""
    diff = np.abs(pred - target)
    if crop:
        diff = diff[crop:-crop, crop:-crop]
        mask = mask[crop:-crop, crop:-crop]
    return float(diff[mask].mean()) if mask.any() else float("nan")


def edge_mask(target: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(target, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(target, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)
    return magnitude >= np.percentile(magnitude, 90.0)


def gradient_magnitude(value: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(value, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(value, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def main(args):
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    dataset = NYUv2FullImageDataset(args.data_root, "test", scale=8, depth_unit_scale=1.0, augment=False)
    selected = set(args.ids or [])
    indices = [i for i, name in enumerate(dataset.names) if not selected or str(name) in selected]
    if not indices:
        raise ValueError("No requested ids matched the 449-image test set")
    all_models = make_models(device)
    requested = [item for item in args.models if item != "gt"]
    models = {name: all_models[name] for name in requested}
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    for directory in (out / "depth_panels", out / "depth_panels" / "individual",
                      out / "depth_error_panels", out / "depth_error_panels" / "individual",
                      out / "gradient_panels", out / "gradient_panels" / "individual",
                      out / "gradient_error_panels", out / "gradient_error_panels" / "individual"):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "data_root": str(args.data_root), "scale": 8, "crop": args.crop,
        "depth_units": "millimeters", "depth_colormap_range": "per-image GT min/max",
        "error_colormap_range_mm": args.error_max_mm,
        "models": requested + ["gt"], "images": len(indices),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    metric_rows = []
    for ds_index in tqdm(indices, desc="RGBDSR visualization", unit="image"):
        depth_lr, rgb, target, dmin, dmax, name = dataset[ds_index]
        depth_lr = depth_lr[None].to(device); rgb = rgb[None].to(device); target = target[None].to(device)
        dmin, dmax = float(dmin), float(dmax)
        target_mm = (target[0, 0].cpu().numpy() * (dmax - dmin) + dmin)
        pred_mm = {}
        for model_name, (kind, origin, rf) in models.items():
            with torch.inference_mode():
                if kind == "coarse_thirdparty":
                    pred = origin(depth_lr, rgb)
                elif kind == "rf":
                    coarse = origin(depth_lr, rgb).clamp(0, 1)
                    pred = rf(rgb, coarse)["final"].clamp(0, 1)
                else:
                    coarse = origin(depth_lr, rgb).clamp(0, 1)
                    pred = rf(rgb, coarse)["final"].clamp(0, 1)
            value = pred[0, 0].cpu().numpy() * (dmax - dmin) + dmin
            pred_mm[model_name] = value

        stem = str(name)
        lo, hi = float(target_mm.min()), float(target_mm.max())
        row = {"name": stem}
        gt_edges = edge_mask(target_mm)
        gt_gradient = gradient_magnitude(target_mm)
        gradient_hi = max(float(np.percentile(gt_gradient, 99.0)), 1e-6)
        depth_tiles = []
        depth_error_tiles = []
        gradient_tiles = []
        gradient_error_tiles = []
        gt_depth_tile = label_tile(resize_tile(colorize(target_mm, lo, hi)), "GT")
        gt_gradient_tile = label_tile(resize_tile(colorize(gt_gradient, 0.0, gradient_hi)), "GT gradient")
        cv2.imwrite(str(out / "depth_panels" / "individual" / f"{stem}_gt.png"), gt_depth_tile)
        cv2.imwrite(str(out / "gradient_panels" / "individual" / f"{stem}_gt.png"), gt_gradient_tile)
        depth_tiles.append(gt_depth_tile)
        gradient_tiles.append(gt_gradient_tile)
        for model_name, value in pred_mm.items():
            rmse, mae = image_metrics(value, target_mm, args.crop)
            row[f"{model_name}_rmse_mm"] = rmse
            row[f"{model_name}_mae_mm"] = mae
            row[f"{model_name}_edge_mae_mm"] = edge_mae(value, target_mm, gt_edges, args.crop)
        metric_rows.append(row)
        for model_name, value in pred_mm.items():
            depth_error = np.abs(value - target_mm)
            pred_gradient = gradient_magnitude(value)
            gradient_error = np.abs(pred_gradient - gt_gradient)
            depth_tile = label_tile(resize_tile(colorize(value, lo, hi)), model_name)
            depth_error_tile = label_tile(
                resize_tile(colorize(depth_error, 0.0, args.error_max_mm)), model_name
            )
            gradient_tile = label_tile(resize_tile(colorize(pred_gradient, 0.0, gradient_hi)), model_name)
            gradient_error_tile = label_tile(resize_tile(colorize(gradient_error, 0.0, gradient_hi)), model_name)
            cv2.imwrite(str(out / "depth_panels" / "individual" / f"{stem}_{model_name}.png"), depth_tile)
            cv2.imwrite(
                str(out / "depth_error_panels" / "individual" / f"{stem}_{model_name}.png"),
                depth_error_tile,
            )
            cv2.imwrite(str(out / "gradient_panels" / "individual" / f"{stem}_{model_name}.png"), gradient_tile)
            cv2.imwrite(str(out / "gradient_error_panels" / "individual" / f"{stem}_{model_name}.png"), gradient_error_tile)
            depth_tiles.append(depth_tile)
            depth_error_tiles.append(depth_error_tile)
            gradient_tiles.append(gradient_tile)
            gradient_error_tiles.append(gradient_error_tile)
        cv2.imwrite(str(out / "depth_panels" / f"{stem}.png"), cv2.hconcat(depth_tiles[1:] + depth_tiles[:1]))
        cv2.imwrite(str(out / "depth_error_panels" / f"{stem}.png"), cv2.hconcat(depth_error_tiles))
        cv2.imwrite(str(out / "gradient_panels" / f"{stem}.png"), cv2.hconcat(gradient_tiles[1:] + gradient_tiles[:1]))
        cv2.imwrite(str(out / "gradient_error_panels" / f"{stem}.png"), cv2.hconcat(gradient_error_tiles))

    if metric_rows:
        fields = ["name"] + [f"{name}_{metric}_mm" for name in requested for metric in ("rmse", "mae", "edge_mae")]
        with (out / "metrics.csv").open("w", newline="") as handle:
            import csv
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(metric_rows)
        summary = {"images": len(metric_rows), "scale": 8, "crop": args.crop,
                   "protocol": "per-image RMSE/MAE then macro-average", "models": requested}
        for name in requested:
            for metric in ("rmse", "mae", "edge_mae"):
                summary[f"{name}_{metric}_mm"] = float(np.nanmean([r[f"{name}_{metric}_mm"] for r in metric_rows]))
                summary[f"{name}_{metric}_cm"] = summary[f"{name}_{metric}_mm"] / 10.0
        (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs" / "x8")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ids", nargs="*", default=None, help="sample ids; omit for all 449 images")
    parser.add_argument("--models", nargs="+", choices=("conv_rf", "dkn", "dctnet", "gt"),
                        default=("conv_rf", "dkn", "dctnet", "gt"))
    parser.add_argument("--crop", type=int, default=6, help="documented evaluation crop; visualization keeps full images")
    parser.add_argument("--error-max-mm", type=float, default=100.0)
    main(parser.parse_args())
