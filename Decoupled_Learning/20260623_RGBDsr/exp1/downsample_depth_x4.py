#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_INPUT_ROOT = Path(r"D:\nuoqi\halo-free\data_NYUv2\NYUv2\depth")
DEFAULT_OUTPUT_ROOT = Path(r"D:\nuoqi\halo-free\data_NYUv2\NYUv2\depth_x4")
VALID_EXTS = {".png", ".tif", ".tiff"}


def iter_depth_files(input_dir: Path):
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def _pil_resample(interpolation: str):
    if interpolation == "area":
        return Image.Resampling.BOX
    if interpolation == "nearest":
        return Image.Resampling.NEAREST
    if interpolation == "bicubic":
        return Image.Resampling.BICUBIC
    raise ValueError(f"Unsupported interpolation: {interpolation}")


def _to_uint16_image(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    if arr.dtype != np.uint16:
        arr = np.clip(arr, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return Image.fromarray(arr, mode="I;16")


def downsample_depth_image(src: Path, dst: Path, scale: int = 4, interpolation: str = "area") -> None:
    if scale <= 0:
        raise ValueError("scale must be positive")

    with Image.open(src) as img:
        width, height = img.size
        out_width = width // scale
        out_height = height // scale
        if out_width <= 0 or out_height <= 0:
            raise ValueError(f"Image is too small for x{scale} downsampling: {src}")

        resized = img.resize((out_width, out_height), resample=_pil_resample(interpolation))
        out = _to_uint16_image(resized)

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)


def process_split(input_root: Path, output_root: Path, split: str, scale: int, interpolation: str, overwrite: bool) -> int:
    input_dir = input_root / split
    output_dir = output_root / split
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing input split directory: {input_dir}")

    count = 0
    for src in iter_depth_files(input_dir):
        dst = output_dir / src.name
        if dst.exists() and not overwrite:
            continue
        downsample_depth_image(src, dst, scale=scale, interpolation=interpolation)
        count += 1
    return count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Downsample NYUv2 depth train/test PNGs by 4x while preserving 16-bit depth values."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument(
        "--interpolation",
        choices=["area", "nearest", "bicubic"],
        default="area",
        help="area uses Pillow BOX resampling.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    total = 0
    for split in args.splits:
        count = process_split(
            args.input_root,
            args.output_root,
            split,
            scale=args.scale,
            interpolation=args.interpolation,
            overwrite=args.overwrite,
        )
        print(f"{split}: wrote {count} files to {args.output_root / split}")
        total += count
    print(f"Total written: {total}")


if __name__ == "__main__":
    main()
