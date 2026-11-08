"""Metrics-only full-image evaluation for Exp B and optional reference summaries."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from exp_b.src.data_exp_b import PairedTripletDataset
    from exp_b.src.model_exp_b import FullExpB
    from exp_b.src.train_exp_b import load_checkpoint, load_full_state
except ImportError:
    from data_exp_b import PairedTripletDataset
    from model_exp_b import FullExpB
    from train_exp_b import load_checkpoint, load_full_state


def trim_even(image: np.ndarray) -> np.ndarray:
    """Crop only the bottom/right odd border, matching previous test protocols."""
    if image.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    height = image.shape[-2] - image.shape[-2] % 2
    width = image.shape[-1] - image.shape[-1] % 2
    if height == 0 or width == 0:
        raise ValueError(f"image is too small for even-border crop: {image.shape}")
    return image[..., :height, :width]


def report_crop_512(image):
    """Take the centered, scale-aligned 512x512 HR report crop."""
    height, width = image.shape[-2:]
    if height < 512 or width < 512:
        raise ValueError(f"test image must be at least 512x512, got {height}x{width}")
    top = ((height - 512) // 2) // 2 * 2
    left = ((width - 512) // 2) // 2 * 2
    return image[..., top : top + 512, left : left + 512]


def report_crop_triplet(lr, hr, gt):
    """Crop aligned 2x LR/HR/GT inputs to the canonical 512 HR field."""
    hr = trim_even(hr)
    gt = trim_even(gt)
    lr = trim_even(lr)
    if hr.shape[-2:] != gt.shape[-2:]:
        raise ValueError(f"HR/GT shape mismatch: {hr.shape} vs {gt.shape}")
    if hr.shape[-2:] != (2 * lr.shape[-2], 2 * lr.shape[-1]):
        raise ValueError(f"expected aligned 2x LR/HR input, got LR={lr.shape}, HR={hr.shape}")
    height, width = hr.shape[-2:]
    if height < 512 or width < 512:
        raise ValueError(f"test image must be at least 512x512 HR, got {height}x{width}")
    top = ((height - 512) // 2) // 2 * 2
    left = ((width - 512) // 2) // 2 * 2
    return (
        lr[..., top // 2 : top // 2 + 256, left // 2 : left // 2 + 256],
        hr[..., top : top + 512, left : left + 512],
        gt[..., top : top + 512, left : left + 512],
    )


def image_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(difference * difference))
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    return {"mse": mse, "psnr_db": psnr}


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not read image: {path}")
    if image.ndim != 2:
        raise ValueError(f"expected single-channel TIFF, got {image.shape}: {path}")
    image = trim_even(np.asarray(image, dtype=np.float32))
    return torch.from_numpy(np.ascontiguousarray(image))[None, None].to(device)


def parse_references(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError("--reference-summary format is NAME=/path/to/summary.json")
        name, path = value.split("=", 1)
        if not name.strip() or not path.strip():
            raise ValueError("--reference-summary requires both a name and a path")
        result.append((name.strip(), Path(path).expanduser()))
    return result


def flatten_reference(name: str, path: Path) -> list[dict]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(summary.get("results"), list):
        rows = []
        for item in summary["results"]:
            rows.append({
                "model": name,
                "output": item.get("output", item.get("model", "final")),
                "mse": item.get("mse", item.get("mean_mse")),
                "mean_psnr_db": item.get("mean_psnr_db", item.get("psnr_db")),
                "source_summary": str(path.resolve()),
            })
        return rows
    mse_by_output = summary.get("mean_of_per_image_mse")
    psnr_by_output = summary.get("mean_psnr_db")
    if isinstance(mse_by_output, dict) or isinstance(psnr_by_output, dict):
        mse_by_output = mse_by_output or {}
        psnr_by_output = psnr_by_output or {}
        output_names = sorted(set(mse_by_output) | set(psnr_by_output))
        return [
            {
                "model": name,
                "output": output,
                "mse": mse_by_output.get(output),
                "mean_psnr_db": psnr_by_output.get(output),
                "source_summary": str(path.resolve()),
            }
            for output in output_names
        ]
    mse = summary.get("mean_image_mse", summary.get("mse"))
    psnr = summary.get("mean_image_psnr_db", summary.get("mean_psnr_db", summary.get("psnr_db")))
    if mse is None and psnr is None:
        raise ValueError(f"unrecognized reference summary format: {path}")
    return [{
        "model": name,
        "output": "final",
        "mse": mse,
        "mean_psnr_db": psnr,
        "source_summary": str(path.resolve()),
    }]


def evaluate_checkpoint(
    model: FullExpB,
    checkpoint_path: Path,
    phase: str,
    dataset: PairedTripletDataset,
    device: torch.device,
    precision: str,
) -> list[dict]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("phase") != phase:
        raise ValueError(f"{checkpoint_path} has phase {checkpoint.get('phase')!r}, expected {phase!r}")
    load_full_state(model, checkpoint)
    model.eval()
    rows = []
    with torch.inference_mode():
        for index, (lr, hr, gt) in enumerate(dataset):
            # The canonical report field is a full 512x512 HR crop with its
            # aligned 256x256 LR input, regardless of source image dimensions.
            lr, hr, gt = report_crop_triplet(lr.numpy(), hr.numpy(), gt.numpy())
            lr_t = torch.from_numpy(np.ascontiguousarray(lr))[None].to(device)
            hr_t = torch.from_numpy(np.ascontiguousarray(hr))[None].to(device)
            gt_t = torch.from_numpy(np.ascontiguousarray(gt))[None].to(device)
            amp = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if precision == "bf16" and device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            with amp:
                outputs = model(lr_t - 0.5, hr_t - 0.5)
            keys = ("coarse0",) if phase == "base" else ("coarse0", "coarse1", "coarse2", "final")
            for output_name in keys:
                prediction = outputs[output_name].float().clamp(0.0, 1.0)[0, 0].cpu().numpy()
                target = gt_t[0, 0].cpu().numpy()
                rows.append({
                    "image_id": dataset.ids[index],
                    "model": phase,
                    "output": output_name,
                    **image_metrics(prediction, target),
                })
            if (index + 1) % 25 == 0 or index + 1 == len(dataset):
                print(f"{phase}: {index + 1}/{len(dataset)}", flush=True)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["output"]), []).append(row)
    summaries = []
    for (model, output), values in sorted(groups.items()):
        summaries.append({
            "model": model,
            "output": output,
            "images": len(values),
            "mean_image_mse": float(np.mean([row["mse"] for row in values])),
            "mean_image_psnr_db": float(np.mean([row["psnr_db"] for row in values])),
        })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/d/ResearchCode/S2R-HDR/data/20260818"),
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--refine-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--reference-summary",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="optional prior result summary; can be supplied multiple times",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for evaluation but is unavailable; pass --device cpu explicitly")
    dataset = PairedTripletDataset(args.data_root, split="test")
    if len(dataset) != 200:
        print(f"Note: test split contains {len(dataset)} images (expected 200).", flush=True)
    model = FullExpB().to(device)
    rows = []
    for phase, path in (
        ("base", args.base_checkpoint),
        ("refine", args.refine_checkpoint),
        ("joint", args.joint_checkpoint),
    ):
        rows.extend(
            evaluate_checkpoint(model, path, phase, dataset, device, args.precision)
        )
    summaries = summarize(rows)
    references = []
    for name, path in parse_references(args.reference_summary):
        if not path.is_file():
            raise FileNotFoundError(f"reference summary not found: {path}")
        references.extend(flatten_reference(name, path))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "per_image_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("image_id", "model", "output", "mse", "psnr_db")
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment": "Exp B VIV-block replacement",
        "data_root": str(args.data_root.resolve()),
        "test_images": len(dataset),
        "inference_geometry": "centered_512x512_hr_crop_aligned_256x256_lr",
        "prediction_images_saved": False,
        "results": summaries,
        "reference_results": references,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
