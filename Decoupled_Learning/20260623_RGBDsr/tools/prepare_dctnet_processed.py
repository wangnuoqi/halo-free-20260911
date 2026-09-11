"""Create DCTNet-style processed NYUv2 data from the local 1000/449 split.

The operation order follows the released DCTNet preprocessing script:
HR depth -> bicubic LR -> HR min/max normalization of LR -> bicubic LR-up;
RGB is converted to [0,1] and ImageNet-normalized. Training samples are
stored in an H5 file; test samples are stored in the official directory layout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def pairs(root: Path, split: str):
    rgb_dir = root / ("RGB_train" if split == "train" else "nyu_rgb_test")
    depth_dir = root / ("depth_train" if split == "train" else "nyu_test_gt_16bit")
    rgb = {p.stem: p for p in rgb_dir.iterdir() if p.is_file()}
    depth = {p.stem: p for p in depth_dir.iterdir() if p.is_file()}
    names = sorted(rgb.keys() & depth.keys(), key=lambda x: int(x) if x.isdigit() else x)
    return [(name, rgb[name], depth[name]) for name in names]


def process(rgb_path: Path, depth_path: Path, scale: int, output_size=None):
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), np.float32)
    if depth_path.suffix.lower() == ".npy":
        depth = np.load(depth_path).astype(np.float32) * 1000.0
    else:
        depth = np.asarray(Image.open(depth_path), np.float32)
    if output_size is not None:
        rgb = np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize((output_size, output_size), Image.Resampling.BICUBIC), np.float32)
        depth = np.asarray(Image.fromarray(depth).resize((output_size, output_size), Image.Resampling.BICUBIC), np.float32)
    h = min(rgb.shape[0], depth.shape[0]); w = min(rgb.shape[1], depth.shape[1])
    h -= h % scale; w -= w % scale
    rgb = rgb[:h, :w].transpose(2, 0, 1) / 255.0
    depth = depth[:h, :w]
    low = np.asarray(Image.fromarray(depth).resize((w // scale, h // scale), Image.Resampling.BICUBIC), np.float32)
    mn, mx = float(depth.min()), float(depth.max())
    low = (low - mn) / max(mx - mn, 1e-6)
    low_up = np.asarray(Image.fromarray(low).resize((w, h), Image.Resampling.BICUBIC), np.float32)[None]
    rgb = (rgb - MEAN) / STD
    return low_up, rgb.astype(np.float32), depth[None].astype(np.float32)


def main(args):
    root = Path(args.data_root)
    train_pairs = pairs(root, "train")
    if args.limit:
        train_pairs = train_pairs[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as h5:
        groups = {key: h5.create_group(key) for key in ("HRDepth", "LRDepth", "RGB")}
        for i, (_, rgb, depth) in enumerate(train_pairs):
            low, image, gt = process(rgb, depth, args.scale, output_size=256)
            groups["HRDepth"].create_dataset(str(i), data=gt)
            groups["LRDepth"].create_dataset(str(i), data=low)
            groups["RGB"].create_dataset(str(i), data=image)
    test_root = Path(args.test_root)
    out = Path(args.test_output)
    for directory in (out / "DepthHR", out / "DepthLrUp", out / "RGB"):
        directory.mkdir(parents=True, exist_ok=True)
    for name, rgb, depth in pairs(test_root, "test"):
        low, image, gt = process(rgb, depth, args.scale)
        np.save(out / "DepthHR" / f"{name}.npy", gt)
        np.save(out / "DepthLrUp" / f"{name}.npy", low)
        np.save(out / "RGB" / f"{name}.npy", image)
    print(f"train={len(train_pairs)} test={len(pairs(test_root, 'test'))} scale={args.scale}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--test-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--test-output", required=True, type=Path)
    p.add_argument("--scale", type=int, choices=(4, 8, 16), default=8)
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
