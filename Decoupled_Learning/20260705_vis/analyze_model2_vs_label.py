import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


DEFAULT_PRED_DIR = Path("/mnt/d/nuoqi/halo-free/Decoupled_Learning/20260705_vis/model_outputs_model2")
DEFAULT_LABEL_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/testlabel")
DEFAULT_OUTPUT_DIR = Path("/mnt/d/nuoqi/halo-free/Decoupled_Learning/20260705_vis/error_analysis_model2")


def prediction_id(path: Path) -> str:
    match = re.match(r"^(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse prediction sample id from {path.name}")
    return match.group(1)


def label_id(path: Path) -> str:
    match = re.match(r"^(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse label sample id from {path.name}")
    return match.group(1)


def match_prediction_and_label(pred_dir: Path, label_dir: Path) -> List[Tuple[str, Path, Path]]:
    pred_map = {prediction_id(p): p for p in sorted(pred_dir.iterdir()) if p.is_file()}
    label_map = {label_id(p): p for p in sorted(label_dir.iterdir()) if p.is_file()}
    common = sorted(set(pred_map) & set(label_map))
    if not common:
        raise ValueError(f"No matching sample ids found between {pred_dir} and {label_dir}")
    return [(sample_id, pred_map[sample_id], label_map[sample_id]) for sample_id in common]


def read_gray(path: Path) -> np.ndarray:
    img = iio.imread(path).astype(np.float32)
    if img.ndim == 3:
        if img.shape[-1] == 1:
            img = img[..., 0]
        else:
            raise ValueError(f"Expected single-channel image, got shape {img.shape} for {path}")
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape} for {path}")
    return img


def conv2d_same(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            out += kernel[y, x] * padded[y : y + img.shape[0], x : x + img.shape[1]]
    return out


def compute_sobel_magnitude(img: np.ndarray) -> np.ndarray:
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = conv2d_same(img.astype(np.float32, copy=False), kx)
    gy = conv2d_same(img.astype(np.float32, copy=False), ky)
    return np.sqrt(gx * gx + gy * gy)


def build_structure_masks(
    grad: np.ndarray,
    strong_percentile: float = 90.0,
    weak_percentile: float = 70.0,
) -> Dict[str, np.ndarray]:
    strong_threshold = float(np.percentile(grad, strong_percentile))
    weak_threshold = float(np.percentile(grad, weak_percentile))
    strong = grad >= strong_threshold
    weak = (grad >= weak_threshold) & (~strong)
    flat = ~(strong | weak)
    return {"strong": strong, "weak": weak, "flat": flat}


def compute_metrics(pred: np.ndarray, label: np.ndarray, eps: float = 1e-6) -> Dict[str, float]:
    diff = pred - label
    abs_diff = np.abs(diff)
    rel_error = abs_diff / (np.abs(label) + eps)

    return {
        "mae": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_error": float(abs_diff.max()),
        "p90_abs_error": float(np.percentile(abs_diff, 90)),
        "p99_abs_error": float(np.percentile(abs_diff, 99)),
        "mean_relative_error": float(rel_error.mean()),
        "median_relative_error": float(np.median(rel_error)),
        "p90_relative_error": float(np.percentile(rel_error, 90)),
        "max_relative_error": float(rel_error.max()),
        "signed_error_mean": float(diff.mean()),
    }


def compute_relative_error(pred: np.ndarray, label: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.abs(pred - label) / (np.abs(label) + eps)


def compute_structure_overlap_stats(
    diff: np.ndarray,
    abs_diff: np.ndarray,
    masks: Dict[str, np.ndarray],
) -> Dict[str, float]:
    pos = diff > 0
    neg = diff < 0
    high_error = abs_diff >= np.percentile(abs_diff, 90)
    stats: Dict[str, float] = {}

    for name, mask in masks.items():
        mask_count = float(mask.sum())
        if mask_count <= 0:
            stats[f"{name}_pixel_ratio"] = 0.0
            stats[f"{name}_mean_abs_error"] = 0.0
            stats[f"{name}_mean_signed_error"] = 0.0
            stats[f"{name}_positive_error_ratio"] = 0.0
            stats[f"{name}_negative_error_ratio"] = 0.0
            stats[f"{name}_high_error_ratio"] = 0.0
            continue

        stats[f"{name}_pixel_ratio"] = mask_count / float(mask.size)
        stats[f"{name}_mean_abs_error"] = float(abs_diff[mask].mean())
        stats[f"{name}_mean_signed_error"] = float(diff[mask].mean())
        stats[f"{name}_positive_error_ratio"] = float((pos & mask).sum() / mask_count)
        stats[f"{name}_negative_error_ratio"] = float((neg & mask).sum() / mask_count)
        stats[f"{name}_high_error_ratio"] = float((high_error & mask).sum() / mask_count)

    return stats


def robust_signed_limit(diff: np.ndarray) -> float:
    return max(abs(float(np.percentile(diff, 1))), abs(float(np.percentile(diff, 99))), 1e-6)


def robust_abs_limit(abs_diff: np.ndarray) -> float:
    return max(float(np.percentile(abs_diff, 99)), 1e-6)


def robust_relative_limit(relative_error: np.ndarray) -> float:
    return max(float(np.percentile(relative_error, 99)), 1e-6)


def save_error_panel(
    path: Path,
    sample_id: str,
    label: np.ndarray,
    diff: np.ndarray,
    abs_diff: np.ndarray,
    relative_error: np.ndarray,
) -> None:
    lim = robust_signed_limit(diff)
    abs_vmax = robust_abs_limit(abs_diff)
    rel_vmax = robust_relative_limit(relative_error)
    label_vmin = float(np.percentile(label, 1))
    label_vmax = float(np.percentile(label, 99))
    if label_vmax <= label_vmin:
        label_vmax = label_vmin + 1e-6

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
    fig.suptitle(f"{sample_id} error analysis")

    specs = [
        (label, "label", "gray", label_vmin, label_vmax),
        (diff, "pred - label", "coolwarm", -lim, lim),
        (abs_diff, "|pred - label|", "magma", 0.0, abs_vmax),
        (relative_error, "|pred-label| / (|label| + eps)", "viridis", 0.0, rel_vmax),
    ]

    for ax, (img, title, cmap, vmin, vmax) in zip(axes, specs):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_structure_panel(
    path: Path,
    sample_id: str,
    label: np.ndarray,
    diff: np.ndarray,
    masks: Dict[str, np.ndarray],
) -> None:
    label_vmin = float(np.percentile(label, 1))
    label_vmax = float(np.percentile(label, 99))
    if label_vmax <= label_vmin:
        label_vmax = label_vmin + 1e-6
    lim = robust_signed_limit(diff)

    structure_rgb = np.zeros((*label.shape, 3), dtype=np.float32)
    structure_rgb[masks["flat"]] = np.array([0.2, 0.2, 0.2], dtype=np.float32)
    structure_rgb[masks["weak"]] = np.array([0.0, 0.9, 1.0], dtype=np.float32)
    structure_rgb[masks["strong"]] = np.array([1.0, 0.9, 0.0], dtype=np.float32)

    overlay = np.zeros((*label.shape, 4), dtype=np.float32)
    overlay[masks["weak"], :3] = np.array([0.0, 0.9, 1.0], dtype=np.float32)
    overlay[masks["strong"], :3] = np.array([1.0, 0.9, 0.0], dtype=np.float32)
    overlay[masks["weak"] | masks["strong"], 3] = 0.28

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    fig.suptitle(f"{sample_id} structure vs signed error")

    axes[0].imshow(label, cmap="gray", vmin=label_vmin, vmax=label_vmax)
    axes[0].set_title("label")
    axes[0].axis("off")

    axes[1].imshow(structure_rgb)
    axes[1].set_title("structure: yellow=strong, cyan=weak, gray=flat")
    axes[1].axis("off")

    im = axes[2].imshow(diff, cmap="coolwarm", vmin=-lim, vmax=lim)
    axes[2].imshow(overlay)
    axes[2].set_title("signed error + structure overlay")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], shrink=0.8)

    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_stats_text(path: Path, sample_id: str, pred_path: Path, label_path: Path, stats: dict[str, float]) -> None:
    lines = [
        f"sample_id: {sample_id}",
        f"prediction: {pred_path}",
        f"label: {label_path}",
    ]
    for key, value in stats.items():
        lines.append(f"{key}: {value:.8g}")
    path.write_text("\n".join(lines) + "\n")


def summarize(rows: List[Dict[str, Union[float, str]]]) -> Dict[str, float]:
    numeric_keys = [key for key in rows[0] if key not in {"sample_id", "prediction_path", "label_path"}]
    out: Dict[str, float] = {}
    for key in numeric_keys:
        values = np.array([float(row[key]) for row in rows], dtype=np.float32)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_median"] = float(np.median(values))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model_outputs_model2 against testlabel and export error maps.")
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vis_dir = args.output_dir / "per_image_vis"
    stats_dir = args.output_dir / "per_image_stats"
    structure_dir = args.output_dir / "per_image_structure"
    vis_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Union[float, str]]] = []
    pairs = match_prediction_and_label(args.pred_dir, args.label_dir)

    for sample_id, pred_path, label_path in tqdm(pairs, desc="Analyzing", dynamic_ncols=True):
        pred = read_gray(pred_path)
        label = read_gray(label_path)
        if pred.shape != label.shape:
            raise ValueError(f"Shape mismatch for {sample_id}: pred={pred.shape}, label={label.shape}")

        diff = pred - label
        abs_diff = np.abs(diff)
        relative_error = compute_relative_error(pred, label)
        grad = compute_sobel_magnitude(label)
        masks = build_structure_masks(grad)
        stats = compute_metrics(pred, label)
        stats.update(compute_structure_overlap_stats(diff, abs_diff, masks))

        save_error_panel(vis_dir / f"{sample_id}_error_panel.png", sample_id, label, diff, abs_diff, relative_error)
        save_structure_panel(structure_dir / f"{sample_id}_structure_overlay.png", sample_id, label, diff, masks)
        write_stats_text(stats_dir / f"{sample_id}_stats.txt", sample_id, pred_path, label_path, stats)

        row: Dict[str, Union[float, str]] = {
            "sample_id": sample_id,
            "prediction_path": str(pred_path),
            "label_path": str(label_path),
        }
        row.update(stats)
        rows.append(row)

    summary_csv = args.output_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_txt = args.output_dir / "summary.txt"
    lines = [f"num_samples: {len(rows)}"]
    for key, value in summary.items():
        lines.append(f"{key}: {value:.8g}")
    summary_txt.write_text("\n".join(lines) + "\n")

    print(f"Wrote {len(rows)} per-image analyses to {args.output_dir}")


if __name__ == "__main__":
    main()
