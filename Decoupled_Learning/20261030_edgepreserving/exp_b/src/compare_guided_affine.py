"""Compare two guided-filter affine reconstructions with fixed Exp B weights."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

from exp_b.src.data_exp_b import (
    PairedPatchCollator,
    PairedTripletDataset,
    deterministic_train_val_indices,
)
from exp_b.src.model_exp_b import FullExpB
from exp_b.src.train_exp_b import load_checkpoint, load_full_state


def affine_fields(self, x_lr, y_lr):
    count = self.box(torch.ones_like(x_lr))
    mean_x = self.box(x_lr) / count
    mean_y = self.box(y_lr) / count
    covariance = self.box(x_lr * y_lr) / count - mean_x * mean_y
    variance = self.box(x_lr * x_lr) / count - mean_x * mean_x
    a = self.a(torch.cat((covariance, variance), dim=1))
    b = mean_y - a * mean_x
    return a, b


def anchored_forward(self, x_lr, y_lr, guide):
    count = self.box(torch.ones_like(x_lr))
    mean_x = self.box(x_lr) / count
    mean_y = self.box(y_lr) / count
    covariance = self.box(x_lr * y_lr) / count - mean_x * mean_y
    variance = self.box(x_lr * x_lr) / count - mean_x * mean_x
    a = self.a(torch.cat((covariance, variance), dim=1))
    size = guide.shape[-2:]
    up = lambda value: F.interpolate(value, size, mode="bilinear", align_corners=True)
    a_hr = up(a)
    b_hr = up(mean_y) - a_hr * up(mean_x)
    return (a_hr * guide + b_hr).clamp(0.0, 1.0)


def edge_aware_resize(field, x_lr, guide_hr, tau):
    """Bilinear weights modulated by LR/HR intensity agreement."""
    batch, _, low_h, low_w = field.shape
    high_h, high_w = guide_hr.shape[-2:]
    y = torch.linspace(0, low_h - 1, high_h, device=field.device, dtype=field.dtype)
    x = torch.linspace(0, low_w - 1, high_w, device=field.device, dtype=field.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    y0, x0 = yy.floor().long(), xx.floor().long()
    y1, x1 = (y0 + 1).clamp_max(low_h - 1), (x0 + 1).clamp_max(low_w - 1)
    wy, wx = yy - y0, xx - x0
    indices = (
        (y0 * low_w + x0, (1 - wy) * (1 - wx)),
        (y0 * low_w + x1, (1 - wy) * wx),
        (y1 * low_w + x0, wy * (1 - wx)),
        (y1 * low_w + x1, wy * wx),
    )
    flat_field = field.flatten(2)
    flat_x = x_lr.flatten(2)
    weighted_values, weights = [], []
    for index, spatial_weight in indices:
        index = index.reshape(1, 1, -1).expand(batch, 1, -1)
        neighbor_field = flat_field.gather(2, index).reshape(batch, 1, high_h, high_w)
        neighbor_x = flat_x.gather(2, index).reshape(batch, 1, high_h, high_w)
        range_weight = torch.exp(-(guide_hr - neighbor_x).abs() / tau)
        weight = spatial_weight[None, None] * range_weight
        weighted_values.append(weight * neighbor_field)
        weights.append(weight)
    return torch.stack(weighted_values).sum(0) / torch.stack(weights).sum(0).clamp_min(1e-12)


def edge_aware_forward(self, x_lr, y_lr, guide, tau):
    a, b = affine_fields(self, x_lr, y_lr)
    a_hr = edge_aware_resize(a, x_lr, guide, tau)
    b_hr = edge_aware_resize(b, x_lr, guide, tau)
    return (a_hr * guide + b_hr).clamp(0.0, 1.0)


def edge_mask(target):
    kernel_x = target.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    magnitude = F.conv2d(target, kernel_x, padding=1).square()
    magnitude += F.conv2d(target, kernel_y, padding=1).square()
    threshold = torch.quantile(magnitude.flatten(), 0.9)
    return magnitude >= threshold


def metrics(prediction, target, edges):
    difference = prediction.float() - target.float()
    mse = difference.square().mean().item()
    return {
        "mse": mse,
        "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
        "bias": difference.mean().item(),
        "mae": difference.abs().mean().item(),
        "edge_mae": difference.abs()[edges].mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--taus", type=float, nargs="+", default=(0.05, 0.1, 0.2))
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    if checkpoint.get("phase") != "joint":
        raise ValueError("expected a joint-phase checkpoint")
    protocol = checkpoint["protocol"]
    device = torch.device(args.device)
    model = FullExpB().to(device)
    load_full_state(model, checkpoint)
    model.eval()
    gf = model.base.gf
    original_forward = gf.forward

    dataset = PairedTripletDataset(args.data_root, split="train")
    _, indices = deterministic_train_val_indices(
        dataset.ids, val_count=protocol["val_count"], seed=protocol["seed"]
    )
    collate = PairedPatchCollator(crop_size=protocol["crop_size"], random_crop=False)
    rows = []
    with torch.inference_mode():
        for position, index in enumerate(indices, 1):
            lr, hr, target = collate([dataset[index]])
            lr, hr, target = (value.to(device) for value in (lr, hr, target))
            edges = edge_mask(target)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=args.precision == "bf16" and device.type == "cuda",
            ):
                baseline = model(lr - 0.5, hr - 0.5)
                variants = [("baseline", baseline)]
                gf.forward = MethodType(anchored_forward, gf)
                try:
                    variants.append(("anchored", model(lr - 0.5, hr - 0.5)))
                finally:
                    gf.forward = original_forward
                for tau in args.taus:
                    gf.forward = MethodType(
                        lambda self, x_lr, y_lr, guide, tau=tau: edge_aware_forward(
                            self, x_lr, y_lr, guide, tau
                        ), gf
                    )
                    try:
                        variants.append((f"edge_aware_tau_{tau:g}", model(lr - 0.5, hr - 0.5)))
                    finally:
                        gf.forward = original_forward
            for output in ("coarse0", "final"):
                for variant, predictions in variants:
                    prediction = predictions[output]
                    rows.append({
                        "image_id": dataset.ids[index],
                        "output": output,
                        "variant": variant,
                        **metrics(prediction, target, edges),
                    })
            if position % 25 == 0 or position == len(indices):
                print(f"evaluated {position}/{len(indices)}", flush=True)

    summary = []
    variants = list(dict.fromkeys(row["variant"] for row in rows))
    for output in ("coarse0", "final"):
        for variant in variants:
            selected = [row for row in rows if row["output"] == output and row["variant"] == variant]
            summary.append({
                "output": output,
                "variant": variant,
                "images": len(selected),
                **{key: sum(row[key] for row in selected) / len(selected)
                   for key in ("mse", "psnr_db", "bias", "mae", "edge_mae")},
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps({
            "checkpoint": str(args.checkpoint.resolve()),
            "protocol": "fixed training validation center crops; no retraining",
            "precision": args.precision,
            "edge_aware_taus": args.taus,
            "results": summary,
        }, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
