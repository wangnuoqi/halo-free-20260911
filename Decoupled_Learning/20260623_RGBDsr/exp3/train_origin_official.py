"""Retrain DKN or DCTNet origin from scratch with the released protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp3.train_exp3_joint import DCTPatchDataset, DKNArrayDataset, TrainableOrigin, unwrap  # noqa: E402


def evaluate(model, loader, device, crop=6):
    model.eval(); rows = []
    with torch.inference_mode():
        for depth, rgb, target, mn, mx in loader:
            depth, rgb, target = depth.to(device), rgb.to(device), target.to(device)
            pred = model(depth, rgb).clamp(0, 1)
            scale = (mx.to(device) - mn.to(device)).view(-1, 1, 1, 1)
            diff = (pred - target) * scale
            if crop:
                diff = diff[..., crop:-crop, crop:-crop]
            for i in range(depth.shape[0]):
                rows.append(float(diff[i].square().mean().sqrt()))
    return float(np.mean(rows))


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.method == "dkn":
        train = DKNArrayDataset(args.dkn_data, True)
        test = DKNArrayDataset(args.dkn_data, False)
        epochs, batch, lr, criterion = 20, 1, 1e-4, nn.L1Loss()
    else:
        train = DCTPatchDataset(args.dct_train)
        test = DKNArrayDataset(args.dkn_data, False)
        epochs, batch, lr, criterion = 200, 8, 1e-3, nn.MSELoss()
    model = TrainableOrigin(args.method, args.init_checkpoint, device) if args.init_checkpoint else TrainableOrigin.from_scratch(args.method)
    model = model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and batch >= torch.cuda.device_count():
        model = nn.DataParallel(model)
    loader = DataLoader(train, batch_size=batch, shuffle=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100 if args.method == "dctnet" else 10000, gamma=0.5 if args.method == "dctnet" else 0.2)
    output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(epochs):
        model.train()
        for depth, rgb, target, *_ in tqdm(loader, desc=f"{args.method} origin {epoch + 1}/{epochs}"):
            depth, rgb, target = depth.to(device), rgb.to(device), target.to(device)
            pred = model(depth, rgb)
            loss = criterion(pred, target)
            optimizer.zero_grad(); loss.backward()
            if args.method == "dctnet": nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        rmse = evaluate(model, test_loader, device, args.crop)
        print(json.dumps({"epoch": epoch + 1, "test_rmse_mm": rmse}))
        state = unwrap(model).state_dict()
        torch.save(state, output / "origin_last.pth")
        if rmse < best:
            best = rmse
            torch.save(state, output / "origin_best.pth")
    summary = {"method": args.method, "epochs": epochs, "batch_size": batch, "best_rmse_cm": best / 10.0,
               "test_images": len(test)}
    (output / "origin_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def _from_scratch(method):
    third = ROOT / "third_party" / ("DKN" if method == "dkn" else "DCTNet")
    sys.path.insert(0, str(third))
    try:
        if method == "dkn":
            from models import DKN
            net = DKN(3, 15, residual=True)
        else:
            from model import DCTNet
            net = DCTNet()
    finally:
        sys.path.pop(0)
    wrapper = TrainableOrigin.__new__(TrainableOrigin)
    nn.Module.__init__(wrapper); wrapper.net = net; wrapper.method = method
    return wrapper


TrainableOrigin.from_scratch = staticmethod(_from_scratch)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("dkn", "dctnet"), required=True)
    p.add_argument("--init-checkpoint", type=Path, default=None)
    p.add_argument("--dkn-data", type=Path, default=ROOT / "exp1/data/dkn_nyuv2_1000_449")
    p.add_argument("--dct-train", type=Path, default=Path(r"D:\nuoqi\halo-free\data_NYUv2\NYUv2-resplit\NYUv2"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--crop", type=int, default=6); p.add_argument("--device", default="cuda:0")
    main(p.parse_args())
