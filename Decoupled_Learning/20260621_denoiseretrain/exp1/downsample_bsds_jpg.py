#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


DEFAULT_DATA_ROOT = Path("/mnt/d/nuoqi/halo-free/data_bsds/traindata_sigma25")


def collect_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}])


def downsample_one(src: Path, dst: Path, scale: int, interpolation: int, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {src}")

    h, w = img.shape[:2]
    new_w = max(1, w // scale)
    new_h = max(1, h // scale)
    down = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), down, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"Failed to write image: {dst}")


def downsample_folder(input_dir: Path, output_dir: Path, scale: int, interpolation: int, overwrite: bool) -> int:
    files = collect_images(input_dir)
    if not files:
        raise FileNotFoundError(f"No JPG/JPEG files found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in tqdm(files, desc=f"{input_dir.name} -> {output_dir.name}", unit="img"):
        downsample_one(src, output_dir / src.name, scale, interpolation, overwrite)
    return len(files)


def parse_args():
    parser = argparse.ArgumentParser(description="Downsample BSDS JPG noisy inputs for ConvGuidedFilter training.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--interpolation",
        choices=["lanczos", "linear", "nearest", "area"],
        default="lanczos",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    interp_map = {
        "lanczos": cv2.INTER_LANCZOS4,
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }
    interpolation = interp_map[args.interpolation]

    train_count = downsample_folder(
        args.data_root / "traininput",
        args.data_root / "traininput_downsample",
        args.scale,
        interpolation,
        args.overwrite,
    )
    test_count = downsample_folder(
        args.data_root / "testinput",
        args.data_root / "testinput_downsample",
        args.scale,
        interpolation,
        args.overwrite,
    )
    print(f"train downsampled: {train_count}")
    print(f"test downsampled: {test_count}")
    print(f"output root: {args.data_root}")


if __name__ == "__main__":
    main()
