#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/test_vivo/testinput")
DEFAULT_LABEL_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/test_vivo/testlabel")
DEFAULT_OUTPUT_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/test_vivo/20260617/convbaseoutput")
DEFAULT_VIS_DIR = Path("/mnt/d/nuoqi/halo-free/Decoupled_Learning/20260621_retrain/analysis/sobel_vis_split")


def common_prefix_id(path: Path) -> str:
    stem = path.stem
    for suffix in ("_pred", "-log", "-WLS-025000"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def build_file_map(folder: Path) -> dict[str, Path]:
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in {".tif", ".tiff"}])
    return {common_prefix_id(p): p for p in files}


def read_gray(path: Path) -> np.ndarray:
    img = iio.imread(path).astype(np.float32)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D or 3D image, got shape {img.shape}: {path}")
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
    img = img.astype(np.float32, copy=False)
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = conv2d_same(img, kx)
    gy = conv2d_same(img, ky)
    return np.sqrt(gx * gx + gy * gy)


def robust_limits(arrays, low=1, high=99) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(vals, low))
    vmax = float(np.percentile(vals, high))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def normalize_with_limits(img: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    scaled = (img.astype(np.float32, copy=False) - vmin) / (vmax - vmin)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def downsample_for_display(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape
    scale = max(h, w) / float(max_side)
    if scale <= 1:
        return img
    step = int(np.ceil(scale))
    return img[::step, ::step]


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=True)
    mask = mask.astype(bool, copy=False)
    padded = np.pad(mask, ((radius, radius), (0, 0)), mode="constant", constant_values=False)
    vertical = np.zeros_like(mask, dtype=bool)
    for offset in range(2 * radius + 1):
        vertical |= padded[offset : offset + mask.shape[0], :]

    padded = np.pad(vertical, ((0, 0), (radius, radius)), mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for offset in range(2 * radius + 1):
        out |= padded[:, offset : offset + mask.shape[1]]
    return out


def mask_for_display(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    shown = img.astype(np.float32, copy=True)
    shown[~mask] = np.nan
    return shown


def save_triplet(path: Path, title: str, images, names, cmap, vmin, vmax):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle(title)
    last = None
    for ax, img, name in zip(axes, images, names):
        last = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
    fig.colorbar(last, ax=axes, shrink=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_single(path: Path, title: str, img: np.ndarray, cmap: str, vmin: float, vmax: float, colorbar: bool = True):
    fig, ax = plt.subplots(1, 1, figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=max(vmax, vmin + 1e-6))
    ax.set_title(title)
    ax.axis("off")
    if colorbar:
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_diff_grid(path: Path, prefix: str, image_diff, grad_diff, abs_image_diff, abs_grad_diff):
    lim_grad = max(abs(float(np.percentile(grad_diff, 1))), abs(float(np.percentile(grad_diff, 99))), 1e-6)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(f"{prefix} gradient-focused difference maps")
    specs = [
        (grad_diff, "sobel(output) - sobel(label)", "coolwarm", -lim_grad, lim_grad),
        (abs_grad_diff, "abs(sobel(output) - sobel(label))", "magma", 0, float(np.percentile(abs_grad_diff, 99))),
        (image_diff, "output - label (auxiliary)", "coolwarm", -max(abs(float(np.percentile(image_diff, 1))), abs(float(np.percentile(image_diff, 99))), 1e-6), max(abs(float(np.percentile(image_diff, 1))), abs(float(np.percentile(image_diff, 99))), 1e-6)),
        (abs_image_diff, "abs(output - label) (auxiliary)", "magma", 0, float(np.percentile(abs_image_diff, 99))),
    ]
    for ax, (img, name, cmap, vmin, vmax) in zip(axes.ravel(), specs):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=max(vmax, vmin + 1e-6))
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_ratio_map(path: Path, prefix: str, ratio: np.ndarray, mask: np.ndarray, title_suffix: str):
    shown = ratio.copy()
    shown[~mask] = np.nan

    fig, ax = plt.subplots(1, 1, figsize=(8, 7), constrained_layout=True)
    fig.suptitle(f"{prefix} sobel(output) / sobel(label), {title_suffix}")
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="0.15")
    im = ax.imshow(shown, cmap=cmap, vmin=0.5, vmax=1.5)
    ax.set_title("blue < 1: output edge weaker; red > 1: stronger")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_signed_error_with_grad_contours(path: Path, prefix: str, value_diff, weak_grad_mask, strong_grad_mask):
    lim = max(abs(float(np.percentile(value_diff, 1))), abs(float(np.percentile(value_diff, 99))), 1e-6)
    fig, ax = plt.subplots(1, 1, figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(value_diff, cmap="coolwarm", vmin=-lim, vmax=lim)
    if np.any(weak_grad_mask):
        ax.contour(weak_grad_mask.astype(np.float32), levels=[0.5], colors=["yellow"], linewidths=0.8)
    if np.any(strong_grad_mask):
        ax.contour(strong_grad_mask.astype(np.float32), levels=[0.5], colors=["lime"], linewidths=0.6)
    ax.set_title(f"{prefix} output-label; yellow=weaker grad, green=stronger grad")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_abs_error_with_grad_overlay(path: Path, prefix: str, abs_value_diff, weak_grad_mask, strong_grad_mask):
    vmax = max(float(np.percentile(abs_value_diff, 99)), 1e-6)
    fig, ax = plt.subplots(1, 1, figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(abs_value_diff, cmap="magma", vmin=0, vmax=vmax)

    overlay = np.zeros((*abs_value_diff.shape, 4), dtype=np.float32)
    overlay[weak_grad_mask] = np.array([0.0, 0.45, 1.0, 0.55], dtype=np.float32)
    overlay[strong_grad_mask] = np.array([0.0, 1.0, 0.2, 0.45], dtype=np.float32)
    ax.imshow(overlay)

    ax.set_title(f"{prefix} abs(output-label); blue=weaker grad, green=stronger grad")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_quadrant_rgb(value_diff, significant_error_mask, weak_grad_mask, strong_grad_mask) -> np.ndarray:
    rgb = np.zeros((*value_diff.shape, 3), dtype=np.float32)
    pos = significant_error_mask & (value_diff > 0)
    neg = significant_error_mask & (value_diff < 0)

    rgb[pos] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rgb[neg] = np.array([0.0, 0.25, 1.0], dtype=np.float32)
    rgb[pos & weak_grad_mask] = np.array([1.0, 0.55, 0.0], dtype=np.float32)
    rgb[neg & weak_grad_mask] = np.array([0.0, 0.9, 1.0], dtype=np.float32)
    rgb[significant_error_mask & strong_grad_mask] = np.array([0.0, 1.0, 0.2], dtype=np.float32)
    return rgb


def save_rgb_map(path: Path, title: str, rgb: np.ndarray):
    fig, ax = plt.subplots(1, 1, figsize=(8, 7), constrained_layout=True)
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_overlay_on_label(path: Path, prefix: str, label_scaled, rgb, alpha_mask):
    fig, ax = plt.subplots(1, 1, figsize=(8, 7), constrained_layout=True)
    ax.imshow(label_scaled, cmap="gray", vmin=0, vmax=1)
    rgba = np.zeros((*label_scaled.shape, 4), dtype=np.float32)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.where(alpha_mask, 0.7, 0.0)
    ax.imshow(rgba)
    ax.set_title(f"{prefix} joint error/gradient classes over label")
    ax.axis("off")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_joint_spatial_maps(
    prefix: str,
    vis_dir: Path,
    max_side: int,
    label_scaled: np.ndarray,
    value_diff: np.ndarray,
    abs_value_diff: np.ndarray,
    grad_ratio: np.ndarray,
    edge_band_mask: np.ndarray,
    ratio_valid_mask: np.ndarray,
    value_error_percentile: float,
):
    edge_abs = abs_value_diff[edge_band_mask]
    if edge_abs.size:
        value_threshold = float(np.percentile(edge_abs, value_error_percentile))
    else:
        value_threshold = float(np.percentile(abs_value_diff, value_error_percentile))

    significant_error_mask = edge_band_mask & (abs_value_diff >= max(value_threshold, 1e-12))
    weak_grad_mask = edge_band_mask & ratio_valid_mask & (grad_ratio < 0.85)
    strong_grad_mask = edge_band_mask & ratio_valid_mask & (grad_ratio > 1.15)

    disp_value_diff = downsample_for_display(value_diff, max_side)
    disp_abs_value_diff = downsample_for_display(abs_value_diff, max_side)
    disp_label = downsample_for_display(label_scaled, max_side)
    disp_sig = downsample_for_display(significant_error_mask, max_side).astype(bool)
    disp_weak = downsample_for_display(weak_grad_mask, max_side).astype(bool)
    disp_strong = downsample_for_display(strong_grad_mask, max_side).astype(bool)

    save_signed_error_with_grad_contours(
        vis_dir / f"{prefix}_signed_error_with_grad_contours.png",
        prefix,
        disp_value_diff,
        disp_weak,
        disp_strong,
    )
    save_abs_error_with_grad_overlay(
        vis_dir / f"{prefix}_abs_error_with_grad_overlay.png",
        prefix,
        disp_abs_value_diff,
        disp_weak,
        disp_strong,
    )

    rgb = make_quadrant_rgb(disp_value_diff, disp_sig, disp_weak, disp_strong)
    save_rgb_map(
        vis_dir / f"{prefix}_edge_joint_quadrant_map.png",
        f"{prefix} edge-band joint map: red/blue=value error; orange/cyan=weaker grad; green=stronger grad",
        rgb,
    )
    save_overlay_on_label(
        vis_dir / f"{prefix}_edge_joint_quadrant_overlay_on_label.png",
        prefix,
        disp_label,
        rgb,
        disp_sig,
    )


def save_split_outputs(
    prefix: str,
    vis_dir: Path,
    max_side: int,
    inp_scaled: np.ndarray,
    output_scaled: np.ndarray,
    label_scaled: np.ndarray,
    grad_inp: np.ndarray,
    grad_output: np.ndarray,
    grad_label: np.ndarray,
    grad_ratio: np.ndarray,
    edge_mask: np.ndarray,
    edge_band_mask: np.ndarray,
    ratio_valid_mask: np.ndarray,
    value_error_percentile: float,
):
    disp_inp = downsample_for_display(inp_scaled, max_side)
    disp_label = downsample_for_display(label_scaled, max_side)
    disp_output = downsample_for_display(output_scaled, max_side)
    disp_grad_inp = downsample_for_display(grad_inp, max_side)
    disp_grad_label = downsample_for_display(grad_label, max_side)
    disp_grad_output = downsample_for_display(grad_output, max_side)
    disp_edge_mask = downsample_for_display(edge_mask, max_side).astype(bool)
    disp_edge_band_mask = downsample_for_display(edge_band_mask, max_side).astype(bool)
    disp_ratio_valid_mask = downsample_for_display(ratio_valid_mask, max_side).astype(bool)

    grad_vmin, grad_vmax = robust_limits([grad_inp, grad_label, grad_output], low=0, high=99)
    grad_diff = grad_output - grad_label
    abs_grad_diff = np.abs(grad_diff)
    value_diff = output_scaled - label_scaled
    abs_value_diff = np.abs(value_diff)
    grad_diff_lim = max(abs(float(np.percentile(grad_diff, 1))), abs(float(np.percentile(grad_diff, 99))), 1e-6)
    value_diff_lim = max(abs(float(np.percentile(value_diff, 1))), abs(float(np.percentile(value_diff, 99))), 1e-6)

    save_single(vis_dir / f"{prefix}_input.png", f"{prefix} input, shared normalized scale", disp_inp, "gray", 0, 1)
    save_single(vis_dir / f"{prefix}_output.png", f"{prefix} output, shared normalized scale", disp_output, "gray", 0, 1)
    save_single(vis_dir / f"{prefix}_label.png", f"{prefix} label, shared normalized scale", disp_label, "gray", 0, 1)

    save_single(vis_dir / f"{prefix}_sobel_input.png", f"{prefix} sobel(input), shared scale", disp_grad_inp, "inferno", grad_vmin, grad_vmax)
    save_single(vis_dir / f"{prefix}_sobel_output.png", f"{prefix} sobel(output), shared scale", disp_grad_output, "inferno", grad_vmin, grad_vmax)
    save_single(vis_dir / f"{prefix}_sobel_label.png", f"{prefix} sobel(label), shared scale", disp_grad_label, "inferno", grad_vmin, grad_vmax)

    save_single(
        vis_dir / f"{prefix}_edge_band_mask.png",
        f"{prefix} label edge band mask",
        disp_edge_band_mask.astype(np.float32),
        "gray",
        0,
        1,
        colorbar=False,
    )
    save_single(
        vis_dir / f"{prefix}_sobel_output_edge_band.png",
        f"{prefix} sobel(output), label edge band only",
        mask_for_display(disp_grad_output, disp_edge_band_mask),
        "inferno",
        grad_vmin,
        grad_vmax,
    )
    save_single(
        vis_dir / f"{prefix}_sobel_label_edge_band.png",
        f"{prefix} sobel(label), label edge band only",
        mask_for_display(disp_grad_label, disp_edge_band_mask),
        "inferno",
        grad_vmin,
        grad_vmax,
    )

    save_ratio_map(
        vis_dir / f"{prefix}_grad_ratio_output_over_label_edges.png",
        prefix,
        downsample_for_display(grad_ratio, max_side),
        disp_edge_mask,
        "strong label edges only",
    )
    save_ratio_map(
        vis_dir / f"{prefix}_grad_ratio_output_over_label_edge_band.png",
        prefix,
        downsample_for_display(grad_ratio, max_side),
        disp_edge_band_mask & disp_ratio_valid_mask,
        "label edge band, valid label-gradient pixels",
    )
    save_single(
        vis_dir / f"{prefix}_grad_diff_output_minus_label.png",
        f"{prefix} sobel(output) - sobel(label)",
        downsample_for_display(grad_diff, max_side),
        "coolwarm",
        -grad_diff_lim,
        grad_diff_lim,
    )
    save_single(
        vis_dir / f"{prefix}_abs_grad_diff_output_minus_label.png",
        f"{prefix} abs(sobel(output) - sobel(label))",
        downsample_for_display(abs_grad_diff, max_side),
        "magma",
        0,
        float(np.percentile(abs_grad_diff, 99)),
    )
    save_single(
        vis_dir / f"{prefix}_value_diff_output_minus_label_aux.png",
        f"{prefix} output - label (auxiliary)",
        downsample_for_display(value_diff, max_side),
        "coolwarm",
        -value_diff_lim,
        value_diff_lim,
    )
    save_single(
        vis_dir / f"{prefix}_abs_value_diff_output_minus_label_aux.png",
        f"{prefix} abs(output - label) (auxiliary)",
        downsample_for_display(abs_value_diff, max_side),
        "magma",
        0,
        float(np.percentile(abs_value_diff, 99)),
    )
    save_joint_spatial_maps(
        prefix,
        vis_dir,
        max_side,
        label_scaled,
        value_diff,
        abs_value_diff,
        grad_ratio,
        edge_band_mask,
        ratio_valid_mask,
        value_error_percentile,
    )


def image_stats(prefix: str, inp, label, output, grad_inp, grad_label, grad_output, edge_mask, edge_band_mask, ratio_valid_mask) -> dict[str, float | str]:
    diff = output - label
    grad_diff = grad_output - grad_label
    edge_ratio = grad_output[edge_mask] / (grad_label[edge_mask] + 1e-12)
    edge_band_ratio_mask = edge_band_mask & ratio_valid_mask
    edge_band_ratio = grad_output[edge_band_ratio_mask] / (grad_label[edge_band_ratio_mask] + 1e-12)
    return {
        "prefix": prefix,
        "input_mean": float(np.mean(inp)),
        "label_mean": float(np.mean(label)),
        "output_mean": float(np.mean(output)),
        "output_minus_label_mean": float(np.mean(diff)),
        "abs_output_minus_label_mean": float(np.mean(np.abs(diff))),
        "grad_input_mean": float(np.mean(grad_inp)),
        "grad_label_mean": float(np.mean(grad_label)),
        "grad_output_mean": float(np.mean(grad_output)),
        "grad_output_minus_label_mean": float(np.mean(grad_diff)),
        "grad_input_p99": float(np.percentile(grad_inp, 99)),
        "grad_label_p99": float(np.percentile(grad_label, 99)),
        "grad_output_p99": float(np.percentile(grad_output, 99)),
        "grad_output_over_label_p99": float(np.percentile(grad_output, 99) / (np.percentile(grad_label, 99) + 1e-12)),
        "label_edge_pixel_count": int(np.count_nonzero(edge_mask)),
        "label_edge_band_pixel_count": int(np.count_nonzero(edge_band_mask)),
        "label_edge_band_ratio_valid_pixel_count": int(np.count_nonzero(edge_band_ratio_mask)),
        "edge_grad_ratio_mean": float(np.mean(edge_ratio)) if edge_ratio.size else float("nan"),
        "edge_grad_ratio_median": float(np.median(edge_ratio)) if edge_ratio.size else float("nan"),
        "edge_grad_ratio_p10": float(np.percentile(edge_ratio, 10)) if edge_ratio.size else float("nan"),
        "edge_grad_ratio_p90": float(np.percentile(edge_ratio, 90)) if edge_ratio.size else float("nan"),
        "edge_band_grad_ratio_mean": float(np.mean(edge_band_ratio)) if edge_band_ratio.size else float("nan"),
        "edge_band_grad_ratio_median": float(np.median(edge_band_ratio)) if edge_band_ratio.size else float("nan"),
        "edge_band_grad_ratio_p10": float(np.percentile(edge_band_ratio, 10)) if edge_band_ratio.size else float("nan"),
        "edge_band_grad_ratio_p90": float(np.percentile(edge_band_ratio, 90)) if edge_band_ratio.size else float("nan"),
    }


def write_stats_text(path: Path, stats: dict[str, float | str]) -> None:
    lines = []
    for key, value in stats.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.8g}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n")


def analyze_one(prefix: str, input_path: Path, label_path: Path, output_path: Path, vis_dir: Path, max_side: int, edge_band_radius: int, value_error_percentile: float):
    inp = read_gray(input_path)
    label = read_gray(label_path)
    output = read_gray(output_path)
    if inp.shape != label.shape or inp.shape != output.shape:
        raise ValueError(f"Shape mismatch for {prefix}: input={inp.shape}, label={label.shape}, output={output.shape}")

    img_vmin, img_vmax = robust_limits([inp, label, output], low=1, high=99)
    inp_scaled = normalize_with_limits(inp, img_vmin, img_vmax)
    label_scaled = normalize_with_limits(label, img_vmin, img_vmax)
    output_scaled = normalize_with_limits(output, img_vmin, img_vmax)

    grad_inp = compute_sobel_magnitude(inp_scaled)
    grad_label = compute_sobel_magnitude(label_scaled)
    grad_output = compute_sobel_magnitude(output_scaled)

    edge_threshold = float(np.percentile(grad_label, 90))
    edge_mask = grad_label > max(edge_threshold, 1e-12)
    edge_band_mask = dilate_mask(edge_mask, edge_band_radius)
    ratio_min_label_grad = max(edge_threshold * 0.1, 1e-6)
    ratio_valid_mask = grad_label > ratio_min_label_grad
    grad_ratio = grad_output / (grad_label + 1e-12)

    save_split_outputs(
        prefix,
        vis_dir,
        max_side,
        inp_scaled,
        output_scaled,
        label_scaled,
        grad_inp,
        grad_output,
        grad_label,
        grad_ratio,
        edge_mask,
        edge_band_mask,
        ratio_valid_mask,
        value_error_percentile,
    )
    stats = image_stats(prefix, inp_scaled, label_scaled, output_scaled, grad_inp, grad_label, grad_output, edge_mask, edge_band_mask, ratio_valid_mask)
    write_stats_text(vis_dir / f"{prefix}_stats.txt", stats)
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize image and Sobel-gradient differences for input/output/label TIFFs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vis-dir", type=Path, default=DEFAULT_VIS_DIR)
    parser.add_argument("--max-side", type=int, default=1600, help="Downsample saved PNG previews to this max side.")
    parser.add_argument("--edge-band-radius", type=int, default=10, help="Dilate label edge mask by this many pixels before band-only visualization.")
    parser.add_argument("--value-error-percentile", type=float, default=90.0, help="Within the edge band, mark abs output-label above this percentile as significant.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.vis_dir.mkdir(parents=True, exist_ok=True)

    input_map = build_file_map(args.input_dir)
    label_map = build_file_map(args.label_dir)
    output_map = build_file_map(args.output_dir)
    prefixes = sorted(set(input_map) & set(label_map) & set(output_map))
    if not prefixes:
        raise SystemExit("No matching TIFF triplets found.")

    rows = []
    for prefix in prefixes:
        print(f"[analyze] {prefix}")
        rows.append(
            analyze_one(
                prefix,
                input_map[prefix],
                label_map[prefix],
                output_map[prefix],
                args.vis_dir,
                args.max_side,
                args.edge_band_radius,
                args.value_error_percentile,
            )
        )

    csv_path = args.vis_dir / "sobel_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} triplets to {args.vis_dir}")


if __name__ == "__main__":
    main()
