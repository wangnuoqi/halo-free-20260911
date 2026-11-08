"""Benchmark the existing 0818 NAFNet and VIVNet checkpoints on test-200."""
from __future__ import annotations

import csv
import argparse
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
DATA_ROOT = Path(r"D:\ResearchCode\S2R-HDR\data\20260818")
BASELINE = PROJECT.parent / "20260815_edgepreserving" / "baseline"
NAFNET_ROOT = BASELINE / "NAFNet"
NAFNET_CHECKPOINT = BASELINE / "runs/nafnet_wls_patch256_nccl_3gpu_bs32/nafnet_wls_best.pth"
VIVNET_CHECKPOINT = BASELINE / "runs/vivnet_wls_patch128_nccl_4gpu_bs48/vivnet_wls_best.pth"
sys.path.insert(0, str(BASELINE.parent))

from baseline.wls_data import WLSPairs, collate_pairs
from baseline.vivnet_wls.model import build_model as build_vivnet, run_padded


def even(x):
    return x[..., :x.shape[-2] - x.shape[-2] % 2,
             :x.shape[-1] - x.shape[-1] % 2]


def make_nafnet(device):
    sys.path.insert(0, str(NAFNET_ROOT))
    from basicsr.models.archs.NAFNet_arch import NAFNetLocal
    return NAFNetLocal(img_channel=1, width=32, enc_blk_nums=[1, 1, 1, 28],
                       middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1],
                       train_size=(1, 1, 256, 256)).to(device)


@torch.inference_mode()
def run_one(model_name, model, predict, dataset, loader, checkpoint, output_dir, device):
    model.eval()
    rows, latencies = [], []
    param_count = sum(p.numel() for p in model.parameters())
    for index, (image, target) in enumerate(loader):
        image, target = even(image).to(device), even(target).to(device)
        if index == 0:
            for _ in range(10):
                predict(model, image)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        prediction = predict(model, image).float().clamp(0, 1)
        torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - start)
        diff = prediction - target.float()
        mse = float(diff.square().mean().cpu())
        mae = float(diff.abs().mean().cpu())
        dx = (prediction[..., :, 1:] - prediction[..., :, :-1]) - (target[..., :, 1:] - target[..., :, :-1])
        dy = (prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
        rows.append({"image_id": dataset.ids[index], "mse": mse, "mae": mae,
                     "psnr_db": 10 * math.log10(1 / max(mse, 1e-12)),
                     "edge_gradient_mae": 0.5 * (dx.abs().mean() + dy.abs().mean()).item()})
        if (index + 1) % 25 == 0 or index + 1 == len(dataset):
            print(f"{model_name} test {index+1}/{len(dataset)}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mean_mse = float(np.mean([r["mse"] for r in rows]))
    summary = {
        "model": model_name, "checkpoint": str(checkpoint.resolve()),
        "test_images": len(rows), "geometry": "full_image_even_crop",
        "precision": "fp32", "parameters": param_count,
        "mean_mse": mean_mse,
        "psnr_from_mean_mse_db": 10 * math.log10(1 / max(mean_mse, 1e-12)),
        "mean_per_image_psnr_db": float(np.mean([r["psnr_db"] for r in rows])),
        "mean_mae": float(np.mean([r["mae"] for r in rows])),
        "mean_edge_gradient_mae": float(np.mean([r["edge_gradient_mae"] for r in rows])),
        "mean_latency_ms": 1000 * float(np.mean(latencies)),
        "p95_latency_ms": 1000 * float(np.percentile(latencies, 95)),
        "device": str(device),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--nafnet-checkpoint", type=Path, default=NAFNET_CHECKPOINT)
    parser.add_argument("--vivnet-checkpoint", type=Path, default=VIVNET_CHECKPOINT)
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT / "expc" / "evaluation" / "test_20260818" / "reference")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    dataset = WLSPairs(args.data_root, "test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_pairs)
    naf_ckpt = torch.load(args.nafnet_checkpoint, map_location="cpu", weights_only=False)
    nafnet = make_nafnet(device)
    nafnet.load_state_dict(naf_ckpt["model_state"], strict=True)
    run_one("NAFNet", nafnet, lambda model, x: model(x), dataset, loader,
            args.nafnet_checkpoint, args.output_root / "nafnet", device)
    del nafnet, naf_ckpt
    torch.cuda.empty_cache()

    viv_ckpt = torch.load(args.vivnet_checkpoint, map_location="cpu", weights_only=False)
    vivnet = build_vivnet(device)
    vivnet.load_state_dict(viv_ckpt["model_state"], strict=True)
    run_one("VIVNet", vivnet, run_padded, dataset, loader, args.vivnet_checkpoint,
            args.output_root / "vivnet", device)


if __name__ == "__main__":
    main()
