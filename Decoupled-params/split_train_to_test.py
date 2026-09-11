#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move matched samples from traininput/trainlabel into testinput/testlabel."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param"),
        help="Dataset root containing traininput and trainlabel.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=700,
        help="Number of matched samples to move from train to test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the selected files without moving them.",
    )
    return parser.parse_args()


def collect_names(directory: Path) -> set[str]:
    return set(os.listdir(directory))


def main() -> None:
    args = parse_args()

    traininput_dir = args.data_root / "traininput"
    trainlabel_dir = args.data_root / "trainlabel"
    testinput_dir = args.data_root / "testinput"
    testlabel_dir = args.data_root / "testlabel"

    for directory in (traininput_dir, trainlabel_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing directory: {directory}")

    traininput_names = collect_names(traininput_dir)
    trainlabel_names = collect_names(trainlabel_dir)

    shared_names = sorted(traininput_names & trainlabel_names)
    input_only = sorted(traininput_names - trainlabel_names)
    label_only = sorted(trainlabel_names - traininput_names)

    if input_only or label_only:
        raise RuntimeError(
            "traininput/trainlabel are not perfectly matched: "
            f"{len(input_only)} only in traininput, {len(label_only)} only in trainlabel."
        )

    if args.count <= 0:
        raise ValueError("--count must be positive.")
    if args.count > len(shared_names):
        raise ValueError(
            f"Requested {args.count} samples, but only {len(shared_names)} matched samples are available."
        )

    testinput_dir.mkdir(exist_ok=True)
    testlabel_dir.mkdir(exist_ok=True)

    existing_test_names = collect_names(testinput_dir) | collect_names(testlabel_dir)
    available_names = [name for name in shared_names if name not in existing_test_names]

    if args.count > len(available_names):
        raise ValueError(
            f"Requested {args.count} new samples, but only {len(available_names)} unmatched train samples remain."
        )

    rng = random.Random(args.seed)
    selected_names = sorted(rng.sample(available_names, args.count))

    print(f"Data root: {args.data_root}")
    print(f"Matched train pairs: {len(shared_names)}")
    print(f"Selected pairs: {len(selected_names)}")
    print(f"Target directories: {testinput_dir} , {testlabel_dir}")

    if args.dry_run:
        print("Dry run enabled. First 10 selected files:")
        for name in selected_names[:10]:
            print(name)
        return

    moved = 0
    for name in selected_names:
        shutil.move(str(traininput_dir / name), str(testinput_dir / name))
        shutil.move(str(trainlabel_dir / name), str(testlabel_dir / name))
        moved += 1

    print(f"Moved {moved} matched pairs from train to test.")


if __name__ == "__main__":
    main()
