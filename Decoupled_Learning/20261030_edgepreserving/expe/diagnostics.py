"""Full-image edge diagnostics and optional single-GPU 4K timing for Exp E."""

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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

PROJECT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT.parent / "20260815_edgepreserving" / "baseline"
sys.path.insert(0, str(BASELINE.parent))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "expc" / "src"))

from baseline.wls_data import WLSPairs, partition_ids
from expe.archive.legacy_a.variants import LEGACY_A_VARIANTS
from expe.model import build_expe_model, otsu_edge_map
from models import build_model
from train_expc import GuidedTriplets, collate_guided


def region_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    """Use GT edges only for evaluation masks, never as model input."""
    prediction, target = prediction.float(), target.float()
    edge = F.max_pool2d(otsu_edge_map(target), 3, stride=1, padding=1).bool()
    flat = ~edge
    difference = (prediction - target).abs()
    mse = (prediction - target).square().mean().item()
    edge_pixels, flat_pixels = edge.sum().item(), flat.sum().item()
    return {
        "mse": mse,
        "psnr_db": 10.0 * math.log10(1.0 / max(mse, 1e-12)),
        "mae": difference.mean().item(),
        "edge_mae": difference[edge].mean().item() if edge_pixels else 0.0,
        "flat_mae": difference[flat].mean().item() if flat_pixels else 0.0,
        "edge_pixels": int(edge_pixels),
        "flat_pixels": int(flat_pixels),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_4k(model, image, lr, device, precision):
    if device.type != "cuda":
        raise ValueError("4K timing requires a CUDA device")
    hr = F.interpolate(image, size=(2160, 3840), mode="bilinear", align_corners=False)
    low = F.interpolate(lr, size=(1080, 1920), mode="bilinear", align_corners=False)
    times = []
    with torch.inference_mode():
        for repetition in range(13):
            _synchronize(device)
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                model(hr, low)
            _synchronize(device)
            if repetition >= 3:
                times.append(time.perf_counter() - start)
    return {"geometry": "resized_to_2160x3840", "warmup": 3, "repeats": 10,
            "mean_ms": 1000 * float(np.mean(times)),
            "p95_ms": 1000 * float(np.percentile(times, 95))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--save-maps", type=int, default=0)
    parser.add_argument("--benchmark-4k", action="store_true")
    parser.add_argument("--oracle-comparison", action="store_true")
    parser.add_argument("--e1-checkpoint", type=Path, default=None,
                        help="Required for old-GF comparisons of non-E1 variants")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    variant = checkpoint["protocol"]["variant"]
    device = torch.device(args.device)
    model = (build_expe_model(variant) if variant in LEGACY_A_VARIANTS
             else build_model(variant)).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    reference = None
    if args.oracle_comparison:
        if variant == "viv_gf_e1":
            reference = model
        else:
            if args.e1_checkpoint is None:
                raise ValueError("--oracle-comparison requires --e1-checkpoint for non-E1 variants")
            baseline_state = torch.load(args.e1_checkpoint, map_location="cpu", weights_only=False)
            if baseline_state.get("protocol", {}).get("variant") != "viv_gf_e1":
                raise ValueError("--e1-checkpoint must contain viv_gf_e1")
            reference = build_model("viv_gf_e1").to(device).eval()
            reference.load_state_dict(baseline_state["model"], strict=True)
    dataset = WLSPairs(args.data_root, "test" if args.split == "test" else "train")
    if args.split == "validation":
        _, validation_ids = partition_ids(dataset.ids, args.val_count, args.seed)
        index = {identifier: position for position, identifier in enumerate(dataset.ids)}
        dataset = Subset(dataset, [index[identifier] for identifier in validation_ids])
    paired = GuidedTriplets(dataset, args.data_root,
                            "test" if args.split == "test" else "train")
    loader = DataLoader(paired, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_guided)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, latencies, benchmark = [], [], None
    for index, (lr, image, target) in enumerate(loader):
        lr = lr.to(device)
        image, target = image.to(device), target.to(device)
        with torch.inference_mode():
            _synchronize(device)
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda" and args.precision == "bf16"):
                prediction, auxiliary = model(image, lr, return_aux=True)
            _synchronize(device)
            latencies.append(time.perf_counter() - start)
            metrics = region_metrics(prediction.clamp(0, 1), target)
            if args.oracle_comparison:
                y_pred = auxiliary["y_lr"]
                y_oracle = F.interpolate(target.float(), size=lr.shape[-2:], mode="area")
                comparisons = {
                    "pred_lr_bilinear": F.interpolate(y_pred, size=image.shape[-2:],
                                                      mode="bilinear", align_corners=True),
                    "pred_lr_old_gf": reference.reconstruct(image, lr, y_pred),
                    "oracle_lr_bilinear": F.interpolate(y_oracle, size=image.shape[-2:],
                                                        mode="bilinear", align_corners=True),
                    "oracle_lr_old_gf": reference.reconstruct(image, lr, y_oracle),
                    "oracle_lr_current": model.reconstruct(image, lr, y_oracle),
                }
                metrics.update({f"{name}_psnr_db": region_metrics(value.clamp(0, 1), target)["psnr_db"]
                                for name, value in comparisons.items()})
            rows.append({"image_id": paired.ids[index], **metrics})
            if index < args.save_maps:
                maps = {key: value[0].float().cpu().numpy()
                        for key, value in auxiliary.items() if isinstance(value, torch.Tensor)}
                maps.update({f"stage_{step}": value[0].float().cpu().numpy()
                             for step, value in enumerate(auxiliary["stages"])})
                np.savez_compressed(args.output_dir / f"maps_{index:03d}.npz", **maps)
            if args.benchmark_4k and benchmark is None:
                benchmark = _benchmark_4k(model, image, lr, device, args.precision)
        if (index + 1) % 25 == 0 or index + 1 == len(loader):
            print(f"{variant} {args.split}: {index + 1}/{len(loader)}", flush=True)

    with (args.output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    edge_pixels = sum(row["edge_pixels"] for row in rows)
    flat_pixels = sum(row["flat_pixels"] for row in rows)
    summary = {
        "variant": variant, "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split, "images": len(rows), "geometry": "full_image_2x_pair",
        "precision": args.precision, "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "mean_per_image_psnr_db": float(np.mean([row["psnr_db"] for row in rows])),
        "psnr_from_mean_mse_db": 10 * math.log10(1 / max(np.mean([row["mse"] for row in rows]), 1e-12)),
        "mean_mae": float(np.mean([row["mae"] for row in rows])),
        "edge_mae": sum(row["edge_mae"] * row["edge_pixels"] for row in rows) / max(edge_pixels, 1),
        "flat_mae": sum(row["flat_mae"] * row["flat_pixels"] for row in rows) / max(flat_pixels, 1),
        "mean_forward_ms": 1000 * float(np.mean(latencies)),
        "four_k_single_gpu": benchmark,
    }
    if args.oracle_comparison:
        summary["oracle_comparison"] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0] if key.endswith("_psnr_db")
        }
        summary["e1_checkpoint"] = (str(args.e1_checkpoint.resolve())
                                    if args.e1_checkpoint else str(args.checkpoint.resolve()))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
