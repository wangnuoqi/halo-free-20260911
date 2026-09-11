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


DEFAULT_INPUT_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/testinput")
DEFAULT_LABEL_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/testlabel")
DEFAULT_OUTPUT_DIR = Path("/mnt/d/nuoqi/halo-free/Decoupled_Learning/20260705_vis/input_label_edge_alignment")
VALID_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def sample_id_from_path(path: Path) -> str:
    match = re.match(r"^(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse sample id from {path.name}")
    return match.group(1)


def is_supported_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES


def match_input_and_label(input_dir: Path, label_dir: Path) -> List[Tuple[str, Path, Path]]:
    input_map = {sample_id_from_path(p): p for p in sorted(input_dir.iterdir()) if is_supported_image_file(p)}
    label_map = {sample_id_from_path(p): p for p in sorted(label_dir.iterdir()) if is_supported_image_file(p)}
    common = sorted(set(input_map) & set(label_map))
    if not common:
        raise ValueError(f"No matching sample ids found between {input_dir} and {label_dir}")
    return [(sample_id, input_map[sample_id], label_map[sample_id]) for sample_id in common]


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


def compute_sobel(img: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = conv2d_same(img.astype(np.float32, copy=False), kx)
    gy = conv2d_same(img.astype(np.float32, copy=False), ky)
    mag = np.sqrt(gx * gx + gy * gy + eps)
    return gx, gy, mag


def otsu_threshold(values: np.ndarray, num_bins: int = 256, eps: float = 1e-6) -> float:
    flat = values.reshape(-1).astype(np.float32)
    vmin = float(flat.min())
    vmax = float(flat.max())
    if vmax - vmin <= eps:
        return vmax
    hist, bin_edges = np.histogram(flat, bins=num_bins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_total = mu[-1]
    denom = np.maximum(omega * (1.0 - omega), eps)
    sigma_between = ((mu_total * omega - mu) ** 2) / denom
    sigma_between[0] = 0.0
    sigma_between[-1] = 0.0
    best_idx = int(np.argmax(sigma_between))
    return float(centers[best_idx])


def build_edge_mask(
    grad_mag: np.ndarray,
    threshold_mode: str,
    threshold_value: float,
) -> Tuple[np.ndarray, float]:
    if threshold_mode == "percentile":
        threshold = float(np.percentile(grad_mag, threshold_value))
    elif threshold_mode == "fixed":
        threshold = float(threshold_value)
    elif threshold_mode == "otsu":
        threshold = otsu_threshold(grad_mag)
    else:
        raise ValueError("threshold_mode must be 'percentile', 'fixed', or 'otsu'")
    mask = grad_mag >= threshold
    if not np.any(mask):
        max_idx = int(np.argmax(grad_mag.reshape(-1)))
        mask = np.zeros_like(grad_mag, dtype=bool)
        mask.reshape(-1)[max_idx] = True
    return mask, threshold


def robust_percentile_vmax(img: np.ndarray, q: float = 99.0, floor: float = 1e-6) -> float:
    return max(float(np.percentile(img, q)), floor)


def robust_gray_range(img: np.ndarray) -> Tuple[float, float]:
    vmin = float(np.percentile(img, 1))
    vmax = float(np.percentile(img, 99))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    count = int(mask.sum())
    if count == 0:
        return 0.0
    return float(values[mask].mean())


def masked_percentile(values: np.ndarray, mask: np.ndarray, q: float) -> float:
    count = int(mask.sum())
    if count == 0:
        return 0.0
    return float(np.percentile(values[mask], q))


def compute_alignment_stats(
    input_mag: np.ndarray,
    label_mag: np.ndarray,
    cosine: np.ndarray,
    input_edge_mask: np.ndarray,
    label_only_edge_mask: np.ndarray,
    reverse_mask: np.ndarray,
) -> Dict[str, float]:
    input_flat_mask = ~input_edge_mask
    mag_ratio = label_mag / np.maximum(input_mag, 1e-6)

    stats: Dict[str, float] = {
        "input_edge_pixel_ratio": float(input_edge_mask.mean()),
        "label_only_edge_pixel_ratio": float(label_only_edge_mask.mean()),
        "reverse_edge_pixel_ratio": float(reverse_mask.mean()),
        "mean_cosine_on_input_edges": masked_mean(cosine, input_edge_mask),
        "median_cosine_on_input_edges": masked_percentile(cosine, input_edge_mask, 50),
        "p10_cosine_on_input_edges": masked_percentile(cosine, input_edge_mask, 10),
        "p90_cosine_on_input_edges": masked_percentile(cosine, input_edge_mask, 90),
        "ratio_cosine_gt_0_8": float(np.mean(cosine[input_edge_mask] > 0.8)) if np.any(input_edge_mask) else 0.0,
        "ratio_cosine_gt_0_9": float(np.mean(cosine[input_edge_mask] > 0.9)) if np.any(input_edge_mask) else 0.0,
        "ratio_cosine_lt_0": float(np.mean(cosine[input_edge_mask] < 0.0)) if np.any(input_edge_mask) else 0.0,
        "ratio_cosine_lt_0_5": float(np.mean(cosine[input_edge_mask] < 0.5)) if np.any(input_edge_mask) else 0.0,
        "mean_mag_ratio_on_input_edges": masked_mean(mag_ratio, input_edge_mask),
        "median_mag_ratio_on_input_edges": masked_percentile(mag_ratio, input_edge_mask, 50),
        "p90_label_grad_on_input_flat": masked_percentile(label_mag, input_flat_mask, 90),
        "mean_label_grad_on_input_flat": masked_mean(label_mag, input_flat_mask),
    }
    return stats


def write_stats_text(
    path: Path,
    sample_id: str,
    input_path: Path,
    label_path: Path,
    input_threshold: float,
    label_threshold: float,
    stats: Dict[str, float],
) -> None:
    lines = [
        f"sample_id: {sample_id}",
        f"input: {input_path}",
        f"label: {label_path}",
        f"input_edge_threshold: {input_threshold:.8g}",
        f"label_edge_threshold: {label_threshold:.8g}",
    ]
    for key, value in stats.items():
        lines.append(f"{key}: {value:.8g}")
    path.write_text("\n".join(lines) + "\n")


def summarize(rows: List[Dict[str, Union[float, str]]]) -> Dict[str, float]:
    numeric_keys = [key for key in rows[0] if key not in {"sample_id", "input_path", "label_path"}]
    out: Dict[str, float] = {}
    for key in numeric_keys:
        values = np.array([float(row[key]) for row in rows], dtype=np.float32)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_median"] = float(np.median(values))
    return out


def save_alignment_panel(
    path: Path,
    sample_id: str,
    input_img: np.ndarray,
    label_img: np.ndarray,
    input_edge_mask: np.ndarray,
    cosine: np.ndarray,
    reverse_mask: np.ndarray,
    label_only_edge_mask: np.ndarray,
) -> None:
    input_vmin, input_vmax = robust_gray_range(input_img)
    label_vmin, label_vmax = robust_gray_range(label_img)

    reverse_overlay = np.zeros((*input_img.shape, 4), dtype=np.float32)
    reverse_overlay[reverse_mask, 0] = 1.0
    reverse_overlay[reverse_mask, 3] = 0.9

    label_only_overlay = np.zeros((*input_img.shape, 4), dtype=np.float32)
    label_only_overlay[label_only_edge_mask, 1] = 1.0
    label_only_overlay[label_only_edge_mask, 3] = 0.9

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(f"{sample_id} input-label edge alignment")

    ax = axes[0, 0]
    ax.imshow(input_img, cmap="gray", vmin=input_vmin, vmax=input_vmax)
    ax.set_title("input")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(label_img, cmap="gray", vmin=label_vmin, vmax=label_vmax)
    ax.set_title("label")
    ax.axis("off")

    ax = axes[0, 2]
    ax.imshow(input_edge_mask.astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title("input edge mask")
    ax.axis("off")

    ax = axes[1, 0]
    im = ax.imshow(cosine, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title("cosine similarity")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 1]
    ax.imshow(input_img, cmap="gray", vmin=input_vmin, vmax=input_vmax)
    ax.imshow(reverse_overlay)
    ax.set_title("reverse-gradient pixels")
    ax.axis("off")

    ax = axes[1, 2]
    ax.imshow(label_img, cmap="gray", vmin=label_vmin, vmax=label_vmax)
    ax.imshow(label_only_overlay)
    ax.set_title("label-only strong edges")
    ax.axis("off")

    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify whether input edges can substitute label edges for edge-preserving alignment."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--threshold-mode",
        choices=["percentile", "fixed", "otsu"],
        default="percentile",
        help="How to build the input edge mask.",
    )
    parser.add_argument(
        "--threshold-value",
        type=float,
        default=90.0,
        help="Percentile value or fixed threshold. Ignored when threshold-mode=otsu.",
    )
    parser.add_argument(
        "--label-strong-percentile",
        type=float,
        default=95.0,
        help="Percentile used to detect label-only strong edges in flat input regions.",
    )
    parser.add_argument(
        "--flat-input-percentile",
        type=float,
        default=30.0,
        help="Input gradient percentile below which the region is treated as flat.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vis_dir = args.output_dir / "per_image_vis"
    stats_dir = args.output_dir / "per_image_stats"
    vis_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Union[float, str]]] = []
    pairs = match_input_and_label(args.input_dir, args.label_dir)

    for sample_id, input_path, label_path in tqdm(pairs, desc="Analyzing", dynamic_ncols=True):
        input_img = read_gray(input_path)
        label_img = read_gray(label_path)
        if input_img.shape != label_img.shape:
            raise ValueError(f"Shape mismatch for {sample_id}: input={input_img.shape}, label={label_img.shape}")

        input_gx, input_gy, input_mag = compute_sobel(input_img)
        label_gx, label_gy, label_mag = compute_sobel(label_img)

        input_edge_mask, input_threshold = build_edge_mask(input_mag, args.threshold_mode, args.threshold_value)
        input_flat_threshold = float(np.percentile(input_mag, args.flat_input_percentile))
        label_strong_threshold = float(np.percentile(label_mag, args.label_strong_percentile))

        dot = input_gx * label_gx + input_gy * label_gy
        denom = np.maximum(input_mag * label_mag, 1e-6)
        cosine = np.clip(dot / denom, -1.0, 1.0)

        reverse_mask = input_edge_mask & (cosine < 0.0)
        input_flat_mask = input_mag <= input_flat_threshold
        label_only_edge_mask = input_flat_mask & (label_mag >= label_strong_threshold)

        stats = compute_alignment_stats(
            input_mag=input_mag,
            label_mag=label_mag,
            cosine=cosine,
            input_edge_mask=input_edge_mask,
            label_only_edge_mask=label_only_edge_mask,
            reverse_mask=reverse_mask,
        )

        save_alignment_panel(
            vis_dir / f"{sample_id}_alignment_panel.png",
            sample_id,
            input_img,
            label_img,
            input_edge_mask,
            cosine,
            reverse_mask,
            label_only_edge_mask,
        )
        write_stats_text(
            stats_dir / f"{sample_id}_stats.txt",
            sample_id,
            input_path,
            label_path,
            input_threshold,
            label_strong_threshold,
            stats,
        )

        row: Dict[str, Union[float, str]] = {
            "sample_id": sample_id,
            "input_path": str(input_path),
            "label_path": str(label_path),
            "input_edge_threshold": input_threshold,
            "label_strong_threshold": label_strong_threshold,
            "input_flat_threshold": input_flat_threshold,
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
    lines = [
        f"num_samples: {len(rows)}",
        f"threshold_mode: {args.threshold_mode}",
        f"threshold_value: {args.threshold_value}",
        f"label_strong_percentile: {args.label_strong_percentile}",
        f"flat_input_percentile: {args.flat_input_percentile}",
    ]
    for key, value in summary.items():
        lines.append(f"{key}: {value:.8g}")
    summary_txt.write_text("\n".join(lines) + "\n")

    print(f"Wrote {len(rows)} input-label edge alignment analyses to {args.output_dir}")


if __name__ == "__main__":
    main()
