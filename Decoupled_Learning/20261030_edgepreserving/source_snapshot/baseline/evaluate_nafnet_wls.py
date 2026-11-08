"""Evaluate a single-channel NAFNet WLS checkpoint on the held-out test set."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
NAFNET_ROOT = ROOT / "baseline" / "NAFNet"
sys.path.insert(0, str(ROOT))

from baseline.wls_data import WLSPairs, collate_pairs


def build_model(device):
    sys.path.insert(0, str(NAFNET_ROOT))
    from basicsr.models.archs.NAFNet_arch import NAFNetLocal

    return NAFNetLocal(
        img_channel=1,
        width=32,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
        train_size=(1, 1, 256, 256),
    ).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path("/mnt/d/ResearchCode/S2R-HDR/data/20260818"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="fp32")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    protocol = checkpoint.get("run_protocol", {})
    if protocol.get("img_channel") != 1:
        raise ValueError("Expected a single-channel NAFNet WLS checkpoint")
    model = build_model(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    dataset = WLSPairs(args.data_root, "test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_pairs)
    rows = []

    with torch.inference_mode():
        for index, (inputs, targets) in enumerate(loader, 1):
            inputs = inputs.to(device)
            targets = targets.to(device).float()
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=args.precision == "bf16" and device.type == "cuda"):
                prediction = model(inputs)
            prediction = prediction.float().clamp(0.0, 1.0)
            error = prediction - targets
            mse = float(error.square().mean().cpu())
            mae = float(error.abs().mean().cpu())
            psnr = float(-10.0 * np.log10(max(mse, 1e-12)))
            identifier = dataset.ids[index - 1]
            rows.append({"id": identifier, "mse": mse, "mae": mae, "psnr_db": psnr})
            if index % 25 == 0 or index == len(dataset):
                print(f"test {index}/{len(dataset)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "mse", "mae", "psnr_db"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": "NAFNetLocal width32 single-channel WLS",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_iter": checkpoint.get("iter"),
        "test_images": len(rows),
        "mse": float(np.mean([row["mse"] for row in rows])),
        "mae": float(np.mean([row["mae"] for row in rows])),
        "mean_psnr_db": float(np.mean([row["psnr_db"] for row in rows])),
        "precision": args.precision,
        "geometry": "full image, batch-minimum even crop",
        "predictions_saved": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
