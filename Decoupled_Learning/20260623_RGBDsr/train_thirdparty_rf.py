"""Train one of the two RF variants on a frozen official x8 DKN/DCTNet coarse model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from exp1.nyuv2_full_dataset import NYUv2FullImageDataset
from thirdparty_refine_models import ThirdPartyCoarse, build_refine, exp6d_loss


def loss_fn(out, target, coarse):
    if getattr(loss_fn, "variant", "original") == "exp6d":
        return exp6d_loss(out, target, coarse)
    delta_gt = target - coarse
    return sum((out[f"delta{i}"] - delta_gt).abs().mean() for i in (1, 2, 3)) / 3.0


def metrics(model, coarse_model, loader, device, crop=6):
    model.eval(); rows = []
    with torch.inference_mode():
        for depth_lr, rgb, target, dmin, dmax, names in loader:
            depth_lr, rgb, target = depth_lr.to(device), rgb.to(device), target.to(device)
            coarse = coarse_model(depth_lr, rgb)
            pred = model(rgb, coarse)["final"].clamp(0, 1)
            scale = (dmax.to(device) - dmin.to(device)).view(-1, 1, 1, 1)
            for i, name in enumerate(names):
                coarse_i, pred_i, target_i = coarse[i:i+1], pred[i:i+1], target[i:i+1]
                if crop:
                    coarse_i, pred_i, target_i = (
                        coarse_i[..., crop:-crop, crop:-crop],
                        pred_i[..., crop:-crop, crop:-crop],
                        target_i[..., crop:-crop, crop:-crop],
                    )
                diff_c = (coarse_i - target_i) * scale[i:i+1]
                diff_p = (pred_i - target_i) * scale[i:i+1]
                rows.append({"name": str(name), "coarse_rmse_mm": float(diff_c.square().mean().sqrt()),
                             "rf_rmse_mm": float(diff_p.square().mean().sqrt()),
                             "coarse_mae_mm": float(diff_c.abs().mean()), "rf_mae_mm": float(diff_p.abs().mean())})
    return rows


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train = NYUv2FullImageDataset(args.data_root, "train", args.scale, indices=range(900), augment=True, depth_unit_scale=1000.0)
    valid = NYUv2FullImageDataset(args.data_root, "train", args.scale, indices=range(900, 1000), depth_unit_scale=1000.0)
    test = NYUv2FullImageDataset(args.test_root, "test", args.scale, depth_unit_scale=1.0)
    train_loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid, 1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, 1, shuffle=False, num_workers=0)
    coarse = ThirdPartyCoarse(
        args.method, args.coarse_checkpoint, device,
        bgr_input=not args.rgb_order,
    )
    model = build_refine(args.variant).to(device)
    loss_fn.variant = args.variant
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if gpu_count > 1 and args.batch_size >= gpu_count:
        coarse = torch.nn.DataParallel(coarse).to(device)
        model = torch.nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        for depth_lr, rgb, target, *_ in tqdm(train_loader, desc=f"{args.method}/{args.variant} {epoch + 1}/{args.epochs}"):
            depth_lr, rgb, target = depth_lr.to(device), rgb.to(device), target.to(device)
            with torch.no_grad():
                coarse_out = coarse(depth_lr, rgb)
            out = model(rgb, coarse_out)
            loss = loss_fn(out, target, coarse_out)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        val_rows = metrics(model, coarse, valid_loader, device, crop=args.crop)
        val_rmse = float(np.mean([x["rf_rmse_mm"] for x in val_rows]))
        print(json.dumps({"epoch": epoch + 1, "val_rf_rmse_mm": val_rmse}))
        if val_rmse < best:
            best = val_rmse
            torch.save((model.module if isinstance(model, torch.nn.DataParallel) else model).state_dict(), args.output_dir / "rf_best.pth")
    target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    target_model.load_state_dict(torch.load(args.output_dir / "rf_best.pth", map_location=device))
    rows = metrics(model, coarse, test_loader, device, crop=args.crop)
    summary = {"method": args.method, "variant": args.variant, "scale": args.scale, "images": len(rows),
               "coarse_rmse_mm": float(np.mean([x["coarse_rmse_mm"] for x in rows])),
               "rf_rmse_mm": float(np.mean([x["rf_rmse_mm"] for x in rows])),
               "coarse_mae_mm": float(np.mean([x["coarse_mae_mm"] for x in rows])),
               "rf_mae_mm": float(np.mean([x["rf_mae_mm"] for x in rows]))}
    (args.output_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--method", choices=("dkn", "dctnet"), required=True)
    parser.add_argument("--variant", choices=("original", "exp4b", "exp6d"), required=True)
    parser.add_argument("--scale", type=int, choices=(8,), default=8)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\nuoqi\halo-free\data_NYUv2\NYUv2-resplit\NYUv2"))
    parser.add_argument("--test-root", type=Path, default=Path(r"D:\nuoqi\halo-free\data_NYUv2\benchmark_test_449"))
    parser.add_argument("--coarse-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80); parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rgb-order", action="store_true", help="keep RGB order for DKN origin")
    main(parser.parse_args())
