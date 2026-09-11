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
DEFAULT_REFERENCE_DIR = Path("/mnt/d/nuoqi/halo-free/Decoupled_Learning/20260705_vis/model_outputs_model2")
DEFAULT_OUTPUT_DIR = Path("/mnt/d/nuoqi/halo-free/Decoupled_Learning/20260705_vis/input_vs_label_transition_alignment")
VALID_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def sample_id_from_path(path: Path) -> str:
    match = re.match(r"^(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse sample id from {path.name}")
    return match.group(1)


def is_supported_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES


def match_input_and_reference(input_dir: Path, reference_dir: Path) -> List[Tuple[str, Path, Path]]:
    input_map = {sample_id_from_path(p): p for p in sorted(input_dir.iterdir()) if is_supported_image_file(p)}
    ref_map = {sample_id_from_path(p): p for p in sorted(reference_dir.iterdir()) if is_supported_image_file(p)}
    common = sorted(set(input_map) & set(ref_map))
    if not common:
        raise ValueError(f"No matching sample ids found between {input_dir} and {reference_dir}")
    return [(sample_id, input_map[sample_id], ref_map[sample_id]) for sample_id in common]


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


def build_edge_mask(grad_mag: np.ndarray, threshold_mode: str, threshold_value: float) -> Tuple[np.ndarray, float]:
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
        idx = int(np.argmax(grad_mag.reshape(-1)))
        mask = np.zeros_like(grad_mag, dtype=bool)
        mask.reshape(-1)[idx] = True
    return mask, threshold


def robust_gray_range(img: np.ndarray) -> Tuple[float, float]:
    vmin = float(np.percentile(img, 1))
    vmax = float(np.percentile(img, 99))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def bilinear_sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = img.shape
    xs = np.clip(xs, 0.0, w - 1.0)
    ys = np.clip(ys, 0.0, h - 1.0)

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = xs - x0
    wy = ys - y0

    top = (1.0 - wx) * img[y0, x0] + wx * img[y0, x1]
    bottom = (1.0 - wx) * img[y1, x0] + wx * img[y1, x1]
    return (1.0 - wy) * top + wy * bottom


def sample_profile(img: np.ndarray, x: float, y: float, nx: float, ny: float, radius: int) -> np.ndarray:
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    xs = x + offsets * nx
    ys = y + offsets * ny
    return bilinear_sample(img, xs, ys).astype(np.float32)


def orient_and_normalize_profiles(
    input_profile: np.ndarray,
    ref_profile: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, float]:
    input_left = float(np.mean(input_profile[:2]))
    input_right = float(np.mean(input_profile[-2:]))
    if input_right < input_left:
        input_profile = input_profile[::-1].copy()
        ref_profile = ref_profile[::-1].copy()
        input_left = float(np.mean(input_profile[:2]))
        input_right = float(np.mean(input_profile[-2:]))

    input_span = float(max(input_right - input_left, eps))
    input_norm = (input_profile - input_left) / input_span
    ref_norm = (ref_profile - input_left) / input_span
    return input_norm, ref_norm, input_span


def profile_metrics(profile_norm: np.ndarray, tol: float = 1e-3) -> Dict[str, float]:
    diff = np.diff(profile_norm)
    return {
        "overshoot": float(max(np.max(profile_norm) - 1.0, 0.0)),
        "undershoot": float(max(0.0 - np.min(profile_norm), 0.0)),
        "excess_variation": float(max(np.sum(np.abs(diff)) - 1.0, 0.0)),
        "negative_slope_ratio": float(np.mean(diff < -tol)),
    }


def select_edge_points(
    core_edge_mask: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = np.argwhere(core_edge_mask)
    if len(candidates) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    if len(candidates) <= max_points:
        return candidates
    indices = rng.choice(len(candidates), size=max_points, replace=False)
    return candidates[indices]


def summarize_numeric_dicts(rows: List[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not rows:
        return out
    keys = list(rows[0].keys())
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=np.float32)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_median"] = float(np.median(values))
    return out


def write_stats_text(
    path: Path,
    sample_id: str,
    input_path: Path,
    reference_path: Path,
    input_threshold: float,
    stats: Dict[str, float],
) -> None:
    lines = [
        f"sample_id: {sample_id}",
        f"input: {input_path}",
        f"reference: {reference_path}",
        f"input_edge_threshold: {input_threshold:.8g}",
    ]
    for key, value in stats.items():
        lines.append(f"{key}: {value:.8g}")
    path.write_text("\n".join(lines) + "\n")


def build_sparse_score_map(shape: Tuple[int, int], points_yx: np.ndarray, scores: np.ndarray) -> np.ndarray:
    score_map = np.full(shape, np.nan, dtype=np.float32)
    if len(points_yx) > 0:
        ys = points_yx[:, 0]
        xs = points_yx[:, 1]
        score_map[ys, xs] = scores.astype(np.float32)
    return score_map


def save_panel(
    path: Path,
    sample_id: str,
    input_img: np.ndarray,
    ref_img: np.ndarray,
    core_edge_mask: np.ndarray,
    score_map: np.ndarray,
    worst_profiles: List[Dict[str, Union[np.ndarray, float, Tuple[int, int]]]],
) -> None:
    input_vmin, input_vmax = robust_gray_range(input_img)
    ref_vmin, ref_vmax = robust_gray_range(ref_img)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(f"{sample_id} edge transition alignment")

    ax = axes[0, 0]
    ax.imshow(input_img, cmap="gray", vmin=input_vmin, vmax=input_vmax)
    ax.set_title("input")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(ref_img, cmap="gray", vmin=ref_vmin, vmax=ref_vmax)
    ax.set_title("reference")
    ax.axis("off")

    ax = axes[0, 2]
    ax.imshow(input_img, cmap="gray", vmin=input_vmin, vmax=input_vmax)
    ax.imshow(core_edge_mask, cmap="Reds", alpha=0.35)
    ax.set_title("input core edge")
    ax.axis("off")

    ax = axes[1, 0]
    im = ax.imshow(score_map, cmap="magma", vmin=0.0, vmax=max(float(np.nanpercentile(score_map, 95)), 1e-6))
    ax.set_title("profile anomaly map")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)

    for idx in range(2):
        ax = axes[1, idx + 1]
        if idx < len(worst_profiles):
            item = worst_profiles[idx]
            input_norm = item["input_norm"]
            ref_norm = item["ref_norm"]
            metric = item["score"]
            y, x = item["point"]
            ax.plot(input_norm, label="input", linewidth=2)
            ax.plot(ref_norm, label="label", linewidth=2)
            ax.set_title(f"worst mismatch #{idx+1} @ ({x}, {y})\nscore={metric:.4f}")
            ax.set_ylim(-0.4, 1.6)
            ax.grid(True, alpha=0.25)
            ax.legend()
        else:
            ax.axis("off")

    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze edge-transition alignment between input and label along input-defined edge normals."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold-mode", choices=["percentile", "fixed", "otsu"], default="otsu")
    parser.add_argument("--threshold-value", type=float, default=90.0)
    parser.add_argument("--profile-radius", type=int, default=4, help="Half-width of the normal profile.")
    parser.add_argument("--max-points-per-image", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vis_dir = args.output_dir / "per_image_vis"
    stats_dir = args.output_dir / "per_image_stats"
    vis_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows: List[Dict[str, Union[float, str]]] = []
    pairs = match_input_and_reference(args.input_dir, args.reference_dir)

    for sample_id, input_path, reference_path in tqdm(pairs, desc="Analyzing", dynamic_ncols=True):
        input_img = read_gray(input_path)
        ref_img = read_gray(reference_path)
        if input_img.shape != ref_img.shape:
            raise ValueError(f"Shape mismatch for {sample_id}: input={input_img.shape}, reference={ref_img.shape}")

        input_gx, input_gy, input_mag = compute_sobel(input_img)
        core_edge_mask, input_threshold = build_edge_mask(input_mag, args.threshold_mode, args.threshold_value)

        points_yx = select_edge_points(core_edge_mask, args.max_points_per_image, rng)
        per_point_rows: List[Dict[str, float]] = []
        worst_profiles: List[Dict[str, Union[np.ndarray, float, Tuple[int, int]]]] = []
        mismatch_scores: List[float] = []

        for y, x in points_yx:
            gx = float(input_gx[y, x])
            gy = float(input_gy[y, x])
            norm = max((gx * gx + gy * gy) ** 0.5, 1e-6)
            nx = gx / norm
            ny = gy / norm

            input_profile = sample_profile(input_img, x, y, nx, ny, args.profile_radius)
            ref_profile = sample_profile(ref_img, x, y, nx, ny, args.profile_radius)
            input_norm, ref_norm, input_span = orient_and_normalize_profiles(
                input_profile,
                ref_profile,
            )
            if input_span <= 1e-5:
                continue

            input_metrics = profile_metrics(input_norm)
            ref_metrics = profile_metrics(ref_norm)
            metrics = {
                "profile_shape_mae": float(np.mean(np.abs(ref_norm - input_norm))),
                "delta_overshoot": ref_metrics["overshoot"] - input_metrics["overshoot"],
                "delta_undershoot": ref_metrics["undershoot"] - input_metrics["undershoot"],
                "delta_excess_variation": ref_metrics["excess_variation"] - input_metrics["excess_variation"],
                "delta_negative_slope_ratio": ref_metrics["negative_slope_ratio"] - input_metrics["negative_slope_ratio"],
                "input_overshoot": input_metrics["overshoot"],
                "input_undershoot": input_metrics["undershoot"],
                "input_excess_variation": input_metrics["excess_variation"],
                "input_negative_slope_ratio": input_metrics["negative_slope_ratio"],
                "label_overshoot": ref_metrics["overshoot"],
                "label_undershoot": ref_metrics["undershoot"],
                "label_excess_variation": ref_metrics["excess_variation"],
                "label_negative_slope_ratio": ref_metrics["negative_slope_ratio"],
                "input_profile_span": input_span,
            }
            per_point_rows.append(metrics)

            mismatch_score = (
                metrics["profile_shape_mae"]
                + 0.5 * abs(metrics["delta_excess_variation"])
                + 0.5 * abs(metrics["delta_overshoot"])
                + 0.5 * abs(metrics["delta_undershoot"])
                + 0.5 * abs(metrics["delta_negative_slope_ratio"])
            )
            mismatch_scores.append(mismatch_score)

            item = {
                "input_norm": input_norm,
                "ref_norm": ref_norm,
                "score": mismatch_score,
                "point": (int(y), int(x)),
            }
            worst_profiles.append(item)

        if not per_point_rows:
            raise ValueError(f"No valid edge profiles found for {sample_id}")

        worst_profiles = sorted(worst_profiles, key=lambda item: float(item["score"]), reverse=True)[:2]
        score_map = build_sparse_score_map(input_img.shape, points_yx[: len(mismatch_scores)], np.array(mismatch_scores, dtype=np.float32))

        per_image = summarize_numeric_dicts(per_point_rows)
        per_image["num_profile_points"] = float(len(per_point_rows))
        per_image["input_edge_pixel_ratio"] = float(core_edge_mask.mean())
        per_image["input_edge_threshold"] = input_threshold

        save_panel(
            vis_dir / f"{sample_id}_profile_panel.png",
            sample_id,
            input_img,
            ref_img,
            core_edge_mask,
            score_map,
            worst_profiles,
        )
        write_stats_text(
            stats_dir / f"{sample_id}_stats.txt",
            sample_id,
            input_path,
            reference_path,
            input_threshold,
            per_image,
        )

        row: Dict[str, Union[float, str]] = {
            "sample_id": sample_id,
            "input_path": str(input_path),
            "reference_path": str(reference_path),
        }
        row.update(per_image)
        rows.append(row)

    summary_csv = args.output_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    numeric_keys = [key for key in rows[0] if key not in {"sample_id", "input_path", "reference_path"}]
    summary_txt = args.output_dir / "summary.txt"
    lines = [
        f"num_samples: {len(rows)}",
        f"threshold_mode: {args.threshold_mode}",
        f"threshold_value: {args.threshold_value}",
        f"profile_radius: {args.profile_radius}",
        f"max_points_per_image: {args.max_points_per_image}",
    ]
    for key in numeric_keys:
        values = np.array([float(row[key]) for row in rows], dtype=np.float32)
        lines.append(f"{key}_mean: {float(values.mean()):.8g}")
        lines.append(f"{key}_median: {float(np.median(values)):.8g}")
    summary_txt.write_text("\n".join(lines) + "\n")

    print(f"Wrote {len(rows)} edge-band profile analyses to {args.output_dir}")


if __name__ == "__main__":
    main()
