"""Evaluate an official third-party x8 coarse checkpoint only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from exp1.nyuv2_full_dataset import NYUv2FullImageDataset
from thirdparty_refine_models import ThirdPartyCoarse


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = NYUv2FullImageDataset(args.test_root, "test", 8, depth_unit_scale=1.0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = ThirdPartyCoarse(
        args.method, args.checkpoint, device,
        clamp_output=not args.no_clamp, bgr_input=not args.rgb_order,
    )
    model.eval()
    rows = []
    with torch.inference_mode():
        for depth_lr, rgb, target, dmin, dmax, names in tqdm(loader, desc=f"{args.method} coarse x8"):
            depth_lr, rgb, target = depth_lr.to(device), rgb.to(device), target.to(device)
            coarse = model(depth_lr, rgb)
            scale = (dmax.to(device) - dmin.to(device)).view(-1, 1, 1, 1)
            diff = (coarse - target) * scale
            if args.crop:
                diff = diff[..., args.crop:-args.crop, args.crop:-args.crop]
            for index, name in enumerate(names):
                image_diff = diff[index]
                rows.append({"name": str(name), "rmse_mm": float(image_diff.square().mean().sqrt()),
                             "mae_mm": float(image_diff.abs().mean())})
    summary = {
        "method": args.method, "scale": 8, "images": len(rows), "crop": args.crop,
        "batch_size": args.batch_size, "bgr": not args.rgb_order,
        "clamp": not args.no_clamp,
        "rmse_mm": float(np.mean([r["rmse_mm"] for r in rows])),
        "rmse_cm": float(np.mean([r["rmse_mm"] for r in rows]) / 10.0),
        "mae_mm": float(np.mean([r["mae_mm"] for r in rows])),
        "mae_cm": float(np.mean([r["mae_mm"] for r in rows]) / 10.0),
        "checkpoint": str(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("dkn", "dctnet"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, default=Path(r"D:\nuoqi\halo-free\data_NYUv2\benchmark_test_449"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rgb-order", action="store_true", help="keep RGB channel order for DKN")
    parser.add_argument("--no-clamp", action="store_true", help="preserve official raw network output")
    main(parser.parse_args())
