#!/usr/bin/env python3
"""Build the array layout consumed by the official DKN NYUv2 loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _paired_paths(data_root: Path):
    pairs = {}
    for split in ("train", "test"):
        rgb_dir = data_root / "image" / split
        depth_dir = data_root / "depth_label" / split
        if not rgb_dir.is_dir() or not depth_dir.is_dir():
            raise FileNotFoundError(f"Missing image/depth_label split under {data_root}: {split}")
        rgb_paths = {path.stem: path for path in rgb_dir.glob("*.png")}
        depth_paths = {path.stem: path for path in depth_dir.glob("*.png")}
        for stem in sorted(rgb_paths.keys() & depth_paths.keys()):
            if not stem.isdigit():
                raise ValueError(f"Expected numeric NYUv2 sample ID, got {stem!r}")
            sample_id = int(stem)
            if sample_id in pairs:
                raise ValueError(f"Duplicate NYUv2 sample ID across splits: {sample_id}")
            pairs[sample_id] = (rgb_paths[stem], depth_paths[stem])

    if not pairs:
        raise RuntimeError(f"No paired image/depth_label files found under {data_root}")
    return pairs


def _load_pair(rgb_path: Path, depth_path: Path):
    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(depth_path) as image:
        depth = np.asarray(image)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = depth.astype(np.float32, copy=False)
    if rgb.shape[:2] != depth.shape:
        raise ValueError(
            f"RGB/depth shape mismatch for {rgb_path.name}: "
            f"RGB={rgb.shape}, depth={depth.shape}"
        )
    if not np.isfinite(depth).all():
        raise ValueError(f"Non-finite depth values found in {depth_path}")
    return rgb, depth


def _load_ids(pairs, ids):
    images = []
    depths = []
    min_values = []
    max_values = []
    expected_shape = None
    for sample_id in ids:
        rgb, depth = _load_pair(*pairs[sample_id])
        if expected_shape is None:
            expected_shape = (rgb.shape, depth.shape)
        elif expected_shape != (rgb.shape, depth.shape):
            raise ValueError(
                f"All samples in a split must have the same shape; sample {sample_id} "
                f"has {(rgb.shape, depth.shape)}, expected {expected_shape}"
            )
        images.append(rgb)
        depths.append(depth)
        min_values.append(float(depth.min()))
        max_values.append(float(depth.max()))
    return (
        np.stack(depths, axis=0).astype(np.float32),
        np.stack(images, axis=0).astype(np.uint8),
        np.asarray(min_values, dtype=np.float32),
        np.asarray(max_values, dtype=np.float32),
    )


def build_dkn_split(data_root, output_root, train_count=1000):
    data_root = Path(data_root)
    output_root = Path(output_root)
    if train_count < 1:
        raise ValueError("train_count must be positive")

    pairs = _paired_paths(data_root)
    all_ids = np.asarray(sorted(pairs), dtype=np.int32)
    if train_count >= len(all_ids):
        raise ValueError(f"train_count={train_count} leaves no test samples from {len(all_ids)} IDs")
    train_ids = all_ids[:train_count]
    test_ids = all_ids[train_count:]

    train_depth, train_images, _, _ = _load_ids(pairs, train_ids)
    test_depth, test_images, test_min, test_max = _load_ids(pairs, test_ids)

    output_root.mkdir(parents=True, exist_ok=True)
    np.save(output_root / "train_depth_split.npy", train_depth)
    np.save(output_root / "train_images_split.npy", train_images)
    np.save(output_root / "test_depth.npy", test_depth)
    np.save(output_root / "test_images_v2.npy", test_images)
    # Explicit semantic order. The official validator's reverse-transform code
    # should be checked before using this auxiliary file for final RMSE.
    np.save(output_root / "test_minmax.npy", np.stack([test_min, test_max], axis=0))
    np.save(output_root / "train_ids.npy", train_ids)
    np.save(output_root / "test_ids.npy", test_ids)

    manifest = {
        "source_root": str(data_root),
        "sample_order": "numeric sample ID ascending across image/train and image/test",
        "total_samples": int(len(all_ids)),
        "train_count": int(len(train_ids)),
        "test_count": int(len(test_ids)),
        "train_id_first_last": [int(train_ids[0]), int(train_ids[-1])],
        "test_id_first_last": [int(test_ids[0]), int(test_ids[-1])],
        "depth_storage": "float32 raw values copied from depth_label PNG",
        "rgb_storage": "uint8 HWC values copied from image PNG",
        "test_minmax_order": "row 0=min, row 1=max",
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {"train_ids": train_ids, "test_ids": test_ids, "manifest": manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=1000)
    args = parser.parse_args()
    result = build_dkn_split(args.data_root, args.output_root, args.train_count)
    print(f"train: {len(result['train_ids'])} samples, IDs {result['train_ids'][0]}..{result['train_ids'][-1]}")
    print(f"test:  {len(result['test_ids'])} samples, IDs {result['test_ids'][0]}..{result['test_ids'][-1]}")
    print(f"wrote: {args.output_root}")


if __name__ == "__main__":
    main()
