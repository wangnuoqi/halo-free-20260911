"""Evaluate one Exp C checkpoint on the untouched full 20260818 test set."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
BASELINE = PROJECT.parent / "20260815_edgepreserving" / "baseline"
sys.path.insert(0, str(BASELINE.parent))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE))
from baseline.wls_data import WLSPairs, collate_pairs
from models import GUIDED_VARIANTS, build_model
from expe.archive.legacy_a.variants import LEGACY_A_VARIANTS
from expe.model import build_expe_model
from train_expc import GuidedTriplets, collate_guided


def even(image):
    return image[..., :image.shape[-2] - image.shape[-2] % 2,
                  :image.shape[-1] - image.shape[-1] % 2]


def report_crop_512(image):
    """Centered, even-aligned 512x512 crop used for every final report."""
    image = even(image)
    height, width = image.shape[-2:]
    if height < 512 or width < 512:
        raise ValueError(f"test image must be at least 512x512, got {height}x{width}")
    top = ((height - 512) // 2) // 2 * 2
    left = ((width - 512) // 2) // 2 * 2
    return image[..., top:top + 512, left:left + 512]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/d/ResearchCode/S2R-HDR/data/20260818"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="fp32")
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    variant = state["protocol"]["variant"]
    device = torch.device(args.device)
    model = (build_expe_model(variant) if variant in LEGACY_A_VARIANTS
             else build_model(variant)).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    base_dataset = WLSPairs(args.data_root, "test")
    guided = variant in GUIDED_VARIANTS or variant in LEGACY_A_VARIANTS
    dataset = GuidedTriplets(base_dataset, args.data_root, "test") if guided else base_dataset
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_guided if guided else collate_pairs)
    rows = []
    latencies = []
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if guided:
                lr, image, target = batch
                lr = even(lr)
                image = even(image)
                target = even(target)
                h, w = image.shape[-2:]
                if h < 512 or w < 512:
                    raise ValueError(f"test image must be at least 512x512, got {h}x{w}")
                top, left = ((h - 512) // 2) // 2 * 2, ((w - 512) // 2) // 2 * 2
                lr = lr[..., top // 2:top // 2 + 256, left // 2:left // 2 + 256].to(device)
                image = image[..., top:top + 512, left:left + 512].to(device)
                target = target[..., top:top + 512, left:left + 512].to(device)
            else:
                image, target = batch
                lr = None
                image = report_crop_512(image).to(device)
                target = report_crop_512(target).to(device)
            if index == 0:
                for _ in range(10):
                    model(image, lr) if lr is not None else model(image)
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=args.precision == "bf16"):
                prediction = model(image, lr) if lr is not None else model(image)
            torch.cuda.synchronize(device)
            latencies.append(time.perf_counter() - start)
            prediction = prediction.float().clamp(0, 1)
            target = target.float()
            diff = prediction - target
            mse = float(diff.square().mean().cpu())
            mae = float(diff.abs().mean().cpu())
            dx = (prediction[..., :, 1:] - prediction[..., :, :-1]) - (target[..., :, 1:] - target[..., :, :-1])
            dy = (prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
            edge_mae = 0.5 * (dx.abs().mean() + dy.abs().mean()).item()
            rows.append({"image_id": base_dataset.ids[index], "mse": mse, "mae": mae,
                         "psnr_db": 10 * math.log10(1 / max(mse, 1e-12)),
                         "edge_gradient_mae": edge_mae})
            if (index + 1) % 25 == 0 or index + 1 == len(dataset):
                print(f"{variant} test {index+1}/{len(dataset)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    mean_mse = float(np.mean([row["mse"] for row in rows]))
    summary = {
        "variant": variant,
        "checkpoint": str(args.checkpoint.resolve()),
        "test_images": len(rows),
        "geometry": "centered_512x512_hr_crop_aligned_256x256_lr",
        "precision": args.precision,
        "parameters": parameter_count,
        "mean_mse": mean_mse,
        "psnr_from_mean_mse_db": 10 * math.log10(1 / max(mean_mse, 1e-12)),
        "mean_per_image_psnr_db": float(np.mean([row["psnr_db"] for row in rows])),
        "mean_mae": float(np.mean([row["mae"] for row in rows])),
        "mean_edge_gradient_mae": float(np.mean([row["edge_gradient_mae"] for row in rows])),
        "mean_latency_ms": 1000 * float(np.mean(latencies)),
        "p95_latency_ms": 1000 * float(np.percentile(latencies, 95)),
        "device": str(device),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
