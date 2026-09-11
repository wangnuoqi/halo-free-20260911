#!/usr/bin/env python3
"""Audit the local NYUv2 RGB-D SR layout without modifying dataset files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def count_files(folder: Path, suffix: str | None = None) -> int:
    if not folder.is_dir():
        return 0
    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and (suffix is None or path.suffix.lower() == suffix)
    )


def first_image(folder: Path):
    files = sorted(path for path in folder.iterdir() if path.is_file())
    return files[0] if files else None


def report_image_folder(folder: Path) -> None:
    path = first_image(folder)
    if path is None:
        print(f"{folder}: empty or missing")
        return
    with Image.open(path) as image:
        print(
            f"{folder}: files={count_files(folder)}, sample={path.name}, "
            f"size={image.size[0]}x{image.size[1]}, mode={image.mode}"
        )


def report_patch_folder(folder: Path) -> None:
    files = sorted(folder.glob("*.npy"))
    if not files:
        print(f"{folder}: empty or missing")
        return
    array = np.load(files[0])
    print(
        f"{folder}: files={len(files)}, sample={files[0].name}, "
        f"shape={array.shape}, dtype={array.dtype}, "
        f"range=[{float(np.nanmin(array)):.6g}, {float(np.nanmax(array)):.6g}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/d/nuoqi/halo-free/data_NYUv2"),
    )
    args = parser.parse_args()
    root = args.data_root
    full_root = root / "NYUv2"

    print("[full-image protocol]")
    for split in ("train", "test"):
        for name in ("image", "depth_label", "depth_x4"):
            report_image_folder(full_root / name / split)

    print("[patch protocol]")
    for scale in ("4X", "8X", "16X"):
        patch_root = root / "benchmark_train_1000" / f"depth_train_{scale}"
        print(f"scale={scale}")
        for name in ("DepthLr", "RGB", "DepthHR"):
            report_patch_folder(patch_root / name)


if __name__ == "__main__":
    main()
