#!/usr/bin/env python3
"""Build traceable direct and synthetic refinement-training samples.

Synthetic samples are converted to the fixed square training resolution before
blending. Reflection images are resized with preserved aspect ratio and
center-cropped to that same resolution.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def progress_iter(iterable: Iterable, total: Optional[int], desc: str):
    """Wrap an iterable with the dataset-generation progress bar."""
    return tqdm(iterable, total=total, desc=desc, unit="sample")


def list_images(directory: Path) -> List[Path]:
    """List images directly inside a known flat dataset directory.

    The extracted DRR image directories are flat.  Recursive traversal is
    deliberately avoided because the archive also contains very large trees
    that are irrelevant to this dataset builder.
    """
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _data_directories(dataset_root: Path, excluded_root: Optional[Path]) -> List[Path]:
    """Return the extracted DRR tree without traversing archives or outputs."""
    excluded = excluded_root.resolve() if excluded_root else None
    candidates = [dataset_root / "DRR" / "DRR"]
    result: List[Path] = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if excluded and (candidate == excluded or excluded in candidate.parents):
            continue
        result.append(candidate)
    return result


def _known_data_dirs(data_root: Path, name: str) -> List[Path]:
    """Return known dataset subdirectories for fast scans on mounted drives."""
    candidates = [
        data_root / "train" / "DRR" / name,
        data_root / "train" / "Syn_dual" / name,
        data_root / "test" / "DRR-C" / name,
        data_root / "test" / "DRR-S" / name,
        data_root / "test" / "Nature" / name,
        data_root / "test" / "real20_420" / name,
        data_root / "test" / "SIR2" / "PostcardDataset" / name,
        data_root / "test" / "SIR2" / "SolidObjectDataset" / name,
        data_root / "test" / "SIR2" / "WildSceneDataset" / name,
        data_root / "test" / "SIR2" / "WildSceneDataset55" / name,
    ]
    return [path for path in candidates if path.is_dir()]


def _same_stem_file(directory: Path, source: Path) -> Optional[Path]:
    for suffix in IMAGE_EXTENSIONS:
        candidate = directory / f"{source.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _dataset_name(path: Path, dataset_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(dataset_root.resolve())
        parts = relative.parts
    except ValueError:
        parts = path.parts
    if "test" in parts:
        index = parts.index("test")
        return parts[index + 1] if index + 1 < len(parts) else "test"
    if "train" in parts:
        index = parts.index("train")
        return parts[index + 1] if index + 1 < len(parts) else "train"
    return "unknown"


def discover_direct_pairs(dataset_root: Path, excluded_root: Optional[Path] = None) -> List[Dict[str, str]]:
    """Discover existing blended/input and transmission/target pairs."""
    records: List[Dict[str, str]] = []
    excluded = excluded_root.resolve() if excluded_root else None

    for data_root in _data_directories(dataset_root, excluded_root):
        blended_dirs = _known_data_dirs(data_root, "blended")
        for blended_dir in blended_dirs:
            if excluded and excluded in blended_dir.resolve().parents:
                continue
            target_dir = blended_dir.parent / "transmission_layer"
            for input_path in list_images(blended_dir):
                target_path = _same_stem_file(target_dir, input_path)
                if target_path:
                    records.append(
                        {
                            "type": "existing",
                            "source_dataset": _dataset_name(input_path, dataset_root),
                            "input": str(input_path.resolve()),
                            "target": str(target_path.resolve()),
                        }
                    )

    syn_root = dataset_root / "DRR" / "DRR" / "train" / "Syn_dual"
    for sample_dir in sorted(syn_root.iterdir()) if syn_root.is_dir() else []:
        if not sample_dir.is_dir() or (excluded and excluded in sample_dir.resolve().parents):
            continue
        target_path = sample_dir / "transmission.png"
        if not target_path.exists():
            continue
        for filename in ("image.png", "image2.png"):
            input_path = sample_dir / filename
            if input_path.exists():
                records.append(
                    {
                        "type": "existing",
                        "source_dataset": "Syn_dual",
                        "input": str(input_path.resolve()),
                        "target": str(target_path.resolve()),
                    }
                )
    return records


def discover_t_pool(dataset_root: Path, excluded_root: Optional[Path] = None) -> List[Path]:
    """Collect clean transmission images usable as synthetic backgrounds."""
    excluded = excluded_root.resolve() if excluded_root else None
    paths: List[Path] = []
    for data_root in _data_directories(dataset_root, excluded_root):
        for directory in _known_data_dirs(data_root, "transmission_layer"):
            if not (excluded and excluded in directory.resolve().parents):
                paths.extend(list_images(directory))
        syn_dir = data_root / "train" / "Syn_dual"
        for path in sorted(syn_dir.glob("*/transmission.png")):
            if not (excluded and excluded in path.resolve().parents):
                paths.append(path)
    return sorted(set(path.resolve() for path in paths))


def discover_r_pool(dataset_root: Path, excluded_root: Optional[Path] = None) -> List[Path]:
    """Collect reflection layers usable as synthetic overlays."""
    excluded = excluded_root.resolve() if excluded_root else None
    paths: List[Path] = []
    for data_root in _data_directories(dataset_root, excluded_root):
        for directory in _known_data_dirs(data_root, "reflection_layer"):
            if not (excluded and excluded in directory.resolve().parents):
                paths.extend(list_images(directory))
        syn_dir = data_root / "train" / "Syn_dual"
        for path in sorted(syn_dir.glob("*/reflection.png")):
            if not (excluded and excluded in path.resolve().parents):
                paths.append(path)
        for path in sorted(syn_dir.glob("*/reflection2.png")):
            if not (excluded and excluded in path.resolve().parents):
                paths.append(path)
    return sorted(set(path.resolve() for path in paths))


def discover_syn_dual_pools(
    dataset_root: Path, excluded_root: Optional[Path] = None
) -> Tuple[List[Path], List[Path]]:
    """Read only the fixed Syn_dual files used for synthetic generation."""
    syn_root = dataset_root / "DRR" / "DRR" / "train" / "Syn_dual"
    if not syn_root.is_dir():
        return [], []

    excluded = excluded_root.resolve() if excluded_root else None
    t_pool: List[Path] = []
    r_pool: List[Path] = []
    sample_dirs = [path for path in syn_root.iterdir() if path.is_dir()]
    for sample_dir in progress_iter(
        sorted(sample_dirs), total=len(sample_dirs), desc="Reading Syn_dual"
    ):
        if excluded and excluded in sample_dir.resolve().parents:
            continue
        transmission = sample_dir / "transmission.png"
        if transmission.is_file():
            t_pool.append(transmission.resolve())
        for filename in ("reflection.png", "reflection2.png"):
            reflection = sample_dir / filename
            if reflection.is_file():
                r_pool.append(reflection.resolve())
    return sorted(set(t_pool)), sorted(set(r_pool))


def fit_to_target(image: Image.Image, target_hw: Tuple[int, int]) -> Image.Image:
    """Resize/crop/pad an image to exact `(height, width)` while preserving aspect ratio."""
    target_h, target_w = target_hw
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target size: {target_hw}")
    image = image.convert("RGB")
    source_w, source_h = image.size
    if source_w <= 0 or source_h <= 0:
        raise ValueError("Input image has no pixels")

    scale = max(target_w / source_w, target_h / source_h)
    resized_w = max(target_w, int(round(source_w * scale)))
    resized_h = max(target_h, int(round(source_h * scale)))
    resized = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    left = max(0, (resized_w - target_w) // 2)
    top = max(0, (resized_h - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h))


def resize_target_pair(
    target: Image.Image, reflection: Image.Image, resolution: int
) -> Tuple[Image.Image, Image.Image]:
    """Convert both source images to the fixed square used by diffusion training."""
    if resolution <= 0:
        raise ValueError(f"Invalid resolution: {resolution}")
    target_fixed = fit_to_target(target, (resolution, resolution))
    reflection_fixed = fit_to_target(reflection, (resolution, resolution))
    return target_fixed, reflection_fixed


def make_synthetic_image(
    target: Image.Image,
    reflection: Image.Image,
    alpha: float,
    beta: float,
) -> Image.Image:
    """Generate the blended image using the paper-style transmission/reflection equation."""
    target = target.convert("RGB")
    reflection = fit_to_target(reflection, (target.height, target.width))
    target_array = np.asarray(target, dtype=np.float32) / 255.0
    reflection_array = np.asarray(reflection, dtype=np.float32) / 255.0
    blended = alpha * target_array + beta * reflection_array - alpha * beta * target_array * reflection_array
    blended = np.clip(blended, 0.0, 1.0)
    return Image.fromarray(np.rint(blended * 255.0).astype(np.uint8))


def make_synthetic_record(
    sample_id: str,
    input_path: Path,
    target_path: Path,
    source_t: Path,
    source_r: Path,
    alpha: float,
    beta: float,
    seed: int,
    width: int,
    height: int,
) -> Dict[str, object]:
    return {
        "id": sample_id,
        "type": "synthetic",
        "input": str(input_path.resolve()),
        "target": str(target_path.resolve()),
        "source_t": str(source_t.resolve()),
        "source_r": str(source_r.resolve()),
        "alpha": float(alpha),
        "beta": float(beta),
        "seed": int(seed),
        "width": int(width),
        "height": int(height),
    }


def _generate_synthetic_sample(task: Tuple[object, ...]) -> Dict[str, object]:
    """Generate one sample; kept top-level so it can be parallelized safely."""
    (
        sample_id,
        input_path,
        target_path,
        source_t,
        source_r,
        alpha,
        beta,
        sample_seed,
        resolution,
    ) = task
    with Image.open(source_t) as target_image, Image.open(source_r) as reflection_image:
        target, reflection = resize_target_pair(target_image, reflection_image, resolution)
        blended = make_synthetic_image(target, reflection, alpha, beta)
        if blended.size != target.size:
            raise AssertionError(f"Generated size mismatch for {sample_id}")
        target.save(target_path)
        blended.save(input_path)
        return make_synthetic_record(
            sample_id,
            input_path,
            target_path,
            source_t,
            source_r,
            alpha,
            beta,
            sample_seed,
            target.width,
            target.height,
        )


def _write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _write_pool(path: Path, values: Sequence[Path]) -> None:
    _write_jsonl(path, ({"path": str(value)} for value in values))


def validate_image_pool(
    values: Sequence[Path], description: str
) -> Tuple[List[Path], List[Dict[str, str]]]:
    """Load each candidate once so a corrupt source cannot stop generation."""
    valid: List[Path] = []
    bad: List[Dict[str, str]] = []
    for path in tqdm(values, total=len(values), desc=f"Validating {description}", unit="image"):
        try:
            with Image.open(path) as image:
                image.load()
            valid.append(path)
        except (OSError, ValueError, SyntaxError) as error:
            bad.append({"path": str(path), "error": str(error)})
    return valid, bad


def build_dataset(
    dataset_root: Path,
    output_root: Path,
    num_synthetic: int,
    seed: int,
    include_direct: bool,
    resolution: int,
    workers: int,
) -> Dict[str, int]:
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # Direct pairs are optional.  Do not inspect the existing blended trees
    # when the caller only requested the synthetic pool.
    direct_records = (
        discover_direct_pairs(dataset_root, output_root) if include_direct else []
    )
    # Synthetic generation only needs the explicit Syn_dual T/R files.  This
    # avoids scanning unrelated DRR test and Blender trees on mounted drives.
    t_pool, r_pool = discover_syn_dual_pools(dataset_root, output_root)
    t_pool, bad_t = validate_image_pool(t_pool, "transmission sources")
    r_pool, bad_r = validate_image_pool(r_pool, "reflection sources")
    bad_sources = bad_t + bad_r
    _write_jsonl(output_root / "bad_sources.jsonl", bad_sources)
    if bad_sources:
        print(f"Skipped {len(bad_sources)} corrupt source image(s); see bad_sources.jsonl")

    if num_synthetic > 0 and (not t_pool or not r_pool):
        raise RuntimeError(f"Need non-empty T/R pools, got T={len(t_pool)} R={len(r_pool)}")

    _write_pool(output_root / "t_pool.jsonl", t_pool)
    _write_pool(output_root / "r_pool.jsonl", r_pool)

    direct_manifest: List[Dict[str, object]] = []
    for index, record in enumerate(
        progress_iter(direct_records, total=len(direct_records), desc="Checking direct pairs")
    ):
        input_path = Path(record["input"])
        target_path = Path(record["target"])
        with Image.open(input_path) as input_image, Image.open(target_path) as target_image:
            if input_image.size != target_image.size:
                raise ValueError(f"Direct pair size mismatch: {input_path} vs {target_path}")
            direct_manifest.append(
                {
                    "id": f"existing_{index:06d}",
                    **record,
                    "width": input_image.width,
                    "height": input_image.height,
                }
            )

    synthetic_dir = output_root / "images" / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    tasks: List[Tuple[object, ...]] = []
    for index in range(num_synthetic):
        sample_seed = rng.randrange(0, 2**31 - 1)
        sample_rng = random.Random(sample_seed)
        source_t = sample_rng.choice(t_pool)
        source_r = sample_rng.choice(r_pool)
        alpha = sample_rng.uniform(0.85, 1.0)
        beta = sample_rng.uniform(0.10, 0.60)
        sample_id = f"synthetic_{index:06d}"
        input_path = synthetic_dir / f"{sample_id}_blended.png"
        target_path = synthetic_dir / f"{sample_id}_transmission.png"
        tasks.append(
            (
                sample_id,
                input_path,
                target_path,
                source_t,
                source_r,
                alpha,
                beta,
                sample_seed,
                resolution,
            )
        )

    synthetic_manifest: List[Dict[str, object]] = []
    if tasks:
        # Threads work well here because Pillow releases the GIL during resize
        # and PNG encoding. Keep only a small bounded queue so the progress bar
        # represents completed files and startup does not enqueue 16k futures.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            next_task = 0
            pending = set()
            queue_size = max(1, workers * 2)

            while next_task < len(tasks) and len(pending) < queue_size:
                pending.add(executor.submit(_generate_synthetic_sample, tasks[next_task]))
                next_task += 1

            completed: Dict[int, Dict[str, object]] = {}
            with tqdm(
                total=len(tasks), desc="Generating synthetic", unit="sample", mininterval=0.5
            ) as progress:
                while pending:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        record = future.result()
                        completed[int(str(record["id"]).split("_")[-1])] = record
                        progress.update(1)
                        if next_task < len(tasks):
                            pending.add(
                                executor.submit(_generate_synthetic_sample, tasks[next_task])
                            )
                            next_task += 1
            synthetic_manifest = [completed[index] for index in range(len(tasks))]

    _write_jsonl(output_root / "direct_manifest.jsonl", direct_manifest)
    _write_jsonl(output_root / "synthetic_manifest.jsonl", synthetic_manifest)
    all_records = (direct_manifest if include_direct else []) + synthetic_manifest
    _write_jsonl(output_root / "manifest.jsonl", all_records)
    return {
        "direct": len(direct_manifest),
        "synthetic": len(synthetic_manifest),
        "total_manifest": len(all_records),
        "t_pool": len(t_pool),
        "r_pool": len(r_pool),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-synthetic", type=int, default=16120)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument(
        "--resolution",
        type=int,
        default=768,
        help="Fixed square size for synthetic input and label images.",
    )
    parser.add_argument("--include-direct", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(16, os.cpu_count() or 1)),
        help="Parallel CPU workers for synthetic generation (default: up to 16).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_synthetic < 0:
        raise ValueError("--num-synthetic must be non-negative")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    counts = build_dataset(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        num_synthetic=args.num_synthetic,
        seed=args.seed,
        include_direct=args.include_direct,
        resolution=args.resolution,
        workers=args.workers,
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
