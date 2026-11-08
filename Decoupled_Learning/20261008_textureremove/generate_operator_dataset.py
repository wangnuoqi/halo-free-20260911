#!/usr/bin/env python3
"""Generate supervised operator-imitation data for ordinary images.

This is the Python counterpart of the old CELiNet/WLS generation flow:
an ordinary image is converted to float data, an operator is evaluated with
sampled parameters, and the input/label pair plus provenance are saved.
Unlike the HDR WLS scripts, this generator does not apply log compression or
per-image log normalization.

The default layout is:

    output/
      train/{input,label}/
      val/{input,label}/
      test/{input,label}/
      metadata.jsonl

For each source image, the input is repeated for every sampled operator
variant. Splits are assigned to source images before variants are generated,
so variants of one image cannot leak across train/val/test.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


try:
    import cv2
except ImportError:  # pragma: no cover - checked at runtime with a clear error
    cv2 = None

try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
except ImportError:  # pragma: no cover - checked at runtime with a clear error
    sparse = None
    spsolve = None


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def require_dependency(name: str, value: Any) -> None:
    if value is None:
        raise RuntimeError(f"{name} is required for this operator; install it before running the generator")


def load_image(path: Path, mode: str) -> np.ndarray:
    image = Image.open(path)
    if mode == "y":
        image = image.convert("YCbCr")
        array = np.asarray(image, dtype=np.float32)[..., 0]
    elif mode == "gray":
        array = np.asarray(image.convert("L"), dtype=np.float32)
    elif mode == "rgb":
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    if array.max() > 1.0:
        array /= 255.0
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def save_float_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        # PIL cannot write arbitrary float RGB arrays as a single image. The
        # generator's default Y mode is single-channel; keep RGB output
        # explicit and lossless if a caller selects --mode rgb.
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(array).save(path, format="TIFF")


def save_input(path: Path, array: np.ndarray, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "tiff":
        save_float_tiff(path.with_suffix(".tiff"), array)
    else:
        uint8 = np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(uint8).save(path.with_suffix(".png"))


def save_label(path: Path, array: np.ndarray, output_format: str) -> None:
    save_input(path, array, output_format)


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(np.float32)


def wls_filter(image: np.ndarray, lambda_: float, alpha: float, epsilon: float = 1e-4) -> np.ndarray:
    """Solve the standard Farbman WLS objective for a grayscale image.

    For RGB input the operator is applied independently to each channel.
    The default CLI mode is Y, matching the single-channel historical WLS
    training data.
    """
    require_dependency("scipy", sparse)
    require_dependency("scipy", spsolve)
    if image.ndim == 3:
        return np.stack(
            [wls_filter(image[..., channel], lambda_, alpha, epsilon) for channel in range(image.shape[-1])],
            axis=-1,
        ).astype(np.float32)
    gray = _gray(image).astype(np.float64)
    height, width = gray.shape
    count = height * width
    # This follows the released MATLAB wlsFilter implementation used by the
    # old CELiNet generator: gradients are computed on log(IN + eps), then
    # affinities are -lambda / (abs(gradient)^alpha + 1e-4). The input itself
    # remains an ordinary [0, 1] image; there is no HDR log normalization.
    guidance = np.log(np.maximum(gray, epsilon))
    weight_l = -lambda_ / (np.abs(np.diff(guidance, axis=1)) ** alpha + 1e-4)
    weight_r = -lambda_ / (np.abs(np.diff(guidance, axis=0)) ** alpha + 1e-4)

    diag = np.ones(count, dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def add_edge(index_a: int, index_b: int, weight: float) -> None:
        diag[index_a] += weight
        diag[index_b] += weight
        rows.extend((index_a, index_b))
        cols.extend((index_b, index_a))
        data.extend((-weight, -weight))

    for y in range(height):
        base = y * width
        for x in range(width - 1):
            add_edge(base + x, base + x + 1, float(-weight_l[y, x]))
    for y in range(height - 1):
        base = y * width
        next_base = (y + 1) * width
        for x in range(width):
            add_edge(base + x, next_base + x, float(-weight_r[y, x]))

    rows.extend(range(count))
    cols.extend(range(count))
    data.extend(diag.tolist())
    system = sparse.coo_matrix((data, (rows, cols)), shape=(count, count)).tocsr()
    result = spsolve(system, gray.reshape(-1)).reshape(height, width)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    require_dependency("opencv-python", cv2)
    kernel = max(3, int(2 * math.ceil(3 * sigma) + 1))
    if image.ndim == 2:
        result = cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    else:
        result = cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def bilateral_filter(image: np.ndarray, diameter: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    require_dependency("opencv-python", cv2)
    result = cv2.bilateralFilter(image, int(diameter), float(sigma_color), float(sigma_space))
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def rolling_guidance_filter(image: np.ndarray, sigma_space: float, sigma_color: float, iterations: int) -> np.ndarray:
    """Practical RGF approximation using repeated bilateral guidance updates."""
    require_dependency("opencv-python", cv2)
    result = image.astype(np.float32, copy=True)
    diameter = max(3, int(2 * math.ceil(3 * sigma_space) + 1))
    for _ in range(int(iterations)):
        result = cv2.bilateralFilter(result, diameter, float(sigma_color) * 255.0, float(sigma_space))
    return np.clip(result, 0.0, 1.0).astype(np.float32)


OPERATORS: dict[str, Callable[..., np.ndarray]] = {
    "wls": wls_filter,
    "gaussian": gaussian_filter,
    "bilateral": bilateral_filter,
    "rolling_guidance": rolling_guidance_filter,
}


DEFAULT_SPECS: dict[str, dict[str, Any]] = {
    "wls": {"count": 7, "lambda_base": 1.0, "lambda_log_range": [-math.log(10), math.log(10)], "alpha_range": [1.0, 1.6]},
    "gaussian": {"count": 7, "sigma_range": [0.8, 4.0]},
    "bilateral": {"count": 7, "diameter": 9, "sigma_color_range": [0.03, 0.20], "sigma_space_range": [2.0, 12.0]},
    "rolling_guidance": {"count": 7, "sigma_space_range": [1.0, 8.0], "sigma_color_range": [0.03, 0.12], "iterations": 4},
}


def sample_parameters(operator: str, spec: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    if operator == "wls":
        lo, hi = spec.get("lambda_log_range", DEFAULT_SPECS[operator]["lambda_log_range"])
        a_lo, a_hi = spec.get("alpha_range", DEFAULT_SPECS[operator]["alpha_range"])
        base = float(spec.get("lambda_base", 1.0))
        return {"lambda_": base * math.exp(rng.uniform(lo, hi)), "alpha": rng.uniform(a_lo, a_hi)}
    if operator == "gaussian":
        lo, hi = spec.get("sigma_range", DEFAULT_SPECS[operator]["sigma_range"])
        return {"sigma": rng.uniform(lo, hi)}
    if operator == "bilateral":
        c_lo, c_hi = spec.get("sigma_color_range", DEFAULT_SPECS[operator]["sigma_color_range"])
        s_lo, s_hi = spec.get("sigma_space_range", DEFAULT_SPECS[operator]["sigma_space_range"])
        return {"diameter": int(spec.get("diameter", 9)), "sigma_color": rng.uniform(c_lo, c_hi), "sigma_space": rng.uniform(s_lo, s_hi)}
    if operator == "rolling_guidance":
        s_lo, s_hi = spec.get("sigma_space_range", DEFAULT_SPECS[operator]["sigma_space_range"])
        c_lo, c_hi = spec.get("sigma_color_range", DEFAULT_SPECS[operator]["sigma_color_range"])
        return {"sigma_space": rng.uniform(s_lo, s_hi), "sigma_color": rng.uniform(c_lo, c_hi), "iterations": int(spec.get("iterations", 4))}
    raise ValueError(f"Unsupported operator: {operator}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--operator", choices=sorted(OPERATORS), default="wls")
    parser.add_argument("--mode", choices=("y", "gray", "rgb"), default="y")
    parser.add_argument("--variants-per-image", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--output-format", choices=("tiff", "png"), default="tiff")
    parser.add_argument("--spec-json", type=Path, help="Optional JSON file overriding operator sampling ranges")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_sources(paths: list[Path], val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[Path]]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1")
    shuffled = sorted(paths)
    random.Random(seed).shuffle(shuffled)
    test_count = int(round(len(shuffled) * test_ratio))
    val_count = int(round(len(shuffled) * val_ratio))
    return {"test": shuffled[:test_count], "val": shuffled[test_count:test_count + val_count], "train": shuffled[test_count + val_count:]}


def main() -> None:
    args = parse_args()
    if args.variants_per_image is not None and args.variants_per_image < 1:
        raise ValueError("variants-per-image must be positive")
    sources = sorted(path for path in args.input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not sources:
        raise RuntimeError(f"No supported images found under {args.input_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {args.output_dir}; pass --overwrite to reuse it")

    spec = dict(DEFAULT_SPECS[args.operator])
    if args.spec_json:
        user_spec = json.loads(args.spec_json.read_text(encoding="utf-8"))
        spec.update(user_spec)
    variants = int(args.variants_per_image or spec.get("count", 7))
    rng = random.Random(args.seed)
    splits = split_sources(sources, args.val_ratio, args.test_ratio, args.seed)
    metadata_path = args.output_dir / "metadata.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("w", encoding="utf-8") as metadata:
        for split, split_sources_list in splits.items():
            for source_path in split_sources_list:
                image = load_image(source_path, args.mode)
                source_id = source_path.relative_to(args.input_dir).with_suffix("").as_posix().replace("/", "__")
                for variant_index in range(variants):
                    params = sample_parameters(args.operator, spec, rng)
                    label = OPERATORS[args.operator](image, **params)
                    stem = f"{source_id}__{args.operator}__v{variant_index:02d}"
                    input_path = args.output_dir / split / "input" / stem
                    label_path = args.output_dir / split / "label" / stem
                    save_input(input_path, image, args.output_format)
                    save_label(label_path, label, args.output_format)
                    metadata.write(json.dumps({
                        "split": split,
                        "source": source_path.relative_to(args.input_dir).as_posix(),
                        "variant": variant_index,
                        "operator": args.operator,
                        "parameters": {"lambda": params["lambda_"], "alpha": params["alpha"]}
                        if args.operator == "wls" else params,
                        "mode": args.mode,
                        "input": str(input_path.with_suffix("." + args.output_format).relative_to(args.output_dir)),
                        "label": str(label_path.with_suffix("." + args.output_format).relative_to(args.output_dir)),
                    }, ensure_ascii=True) + "\n")
    print(f"Generated {len(sources) * variants} pairs from {len(sources)} source images")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
