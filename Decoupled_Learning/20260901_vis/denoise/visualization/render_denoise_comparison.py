"""Render BSD68 denoising and edge-preservation comparisons."""
from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = HERE / "exp8a_vs_official"
METHODS = OrderedDict(
    [
        ("ours", ("Ours: ConvGuided + RF", "final")),
        ("ffdnet", ("Official FFDNet", "ffdnet_official_sigma25/coarse")),
        ("dncnn", ("Official DnCNN", "dncnn_official_sigma25/coarse")),
    ]
)
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
RATIO_VMIN = 0.5
RATIO_VMAX = 1.5


def collect_files(root: Path) -> Dict[str, Path]:
    if not root.is_dir():
        raise RuntimeError(f"Missing result directory: {root}")
    return {
        item.stem: item
        for item in sorted(root.iterdir())
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    }


def read_float(path: Path) -> np.ndarray:
    value = np.asarray(imageio.imread(path))
    if value.ndim == 3:
        if value.shape[-1] == 1:
            value = value[..., 0]
        else:
            value = cv2.cvtColor(value, cv2.COLOR_RGB2GRAY)
    value = value.astype(np.float32, copy=False)
    if float(value.max(initial=0.0)) > 1.0:
        value = value / 255.0
    return value


def psnr(output: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    mse = float(np.mean((output - target) ** 2))
    score = 100.0 if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    return score, mse


def save_figure(figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def sobel_components(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)
    return gx, gy, magnitude


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    size = radius * 2 + 1
    kernel = np.ones((size, size), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def shared_percentile(values: Iterable[np.ndarray], percentile: float) -> float:
    limits = []
    for value in values:
        finite = value[np.isfinite(value)]
        if finite.size:
            limits.append(float(np.percentile(finite, percentile)))
    return max(max(limits, default=1.0), 1e-8)


def render_four_panel(
    sample_id: str,
    images: OrderedDict,
    scores: Dict[str, float],
    output: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(17, 4.8), constrained_layout=True)
    items = list(images.items())
    for axis, (key, image) in zip(axes, items):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_axis_off()
        title = "GT" if key == "gt" else f"{METHODS[key][0]}\n{scores[key]:.2f} dB"
        axis.set_title(title, fontsize=11, pad=8)
    figure.suptitle(f"BSD68 {sample_id}: denoising comparison", fontsize=13)
    save_figure(figure, output / f"{sample_id}_four_panel.png", dpi)


def render_single_images(
    sample_id: str,
    images: OrderedDict,
    scores: Dict[str, float],
    output: Path,
    dpi: int,
) -> None:
    for key, image in images.items():
        figure, axis = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_axis_off()
        title = "GT" if key == "gt" else f"{METHODS[key][0]} | {scores[key]:.2f} dB"
        axis.set_title(f"BSD68 {sample_id}: {title}", fontsize=12, pad=8)
        save_figure(figure, output / f"{sample_id}_{key}.png", dpi)


def render_error_heatmap(
    sample_id: str,
    images: OrderedDict,
    target: np.ndarray,
    scores: Dict[str, float],
    error_vmax: float,
    output: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.9), constrained_layout=True)
    rendered = None
    for axis, key in zip(axes, METHODS):
        error = np.abs(images[key] - target)
        rendered = axis.imshow(error, cmap="magma", vmin=0.0, vmax=error_vmax)
        axis.set_axis_off()
        axis.set_title(f"{METHODS[key][0]} error\n{scores[key]:.2f} dB", fontsize=11, pad=8)
    colorbar = figure.colorbar(rendered, ax=axes.tolist(), fraction=0.025, pad=0.02)
    colorbar.set_label("|output - GT|")
    figure.suptitle(f"BSD68 {sample_id}: absolute error heatmaps", fontsize=13)
    save_figure(figure, output / f"{sample_id}_error_heatmap.png", dpi)


def render_edge_summary(
    sample_id: str,
    images: OrderedDict,
    strong_edge_ratios: OrderedDict,
    strong_edge_mask: np.ndarray,
    scores: Dict[str, float],
    output: Path,
    dpi: int,
) -> None:
    colormap = plt.get_cmap("coolwarm").copy()
    colormap.set_bad((0.12, 0.12, 0.12, 1.0))
    figure, axes = plt.subplots(2, 4, figsize=(17, 9), constrained_layout=True)
    display_items = list(METHODS.items()) + [("gt", ("GT", ""))]

    for axis, (key, _) in zip(axes[0], display_items):
        image = images[key]
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_axis_off()
        title = "GT" if key == "gt" else f"{METHODS[key][0]}\n{scores[key]:.2f} dB"
        axis.set_title(title, fontsize=11, pad=8)

    for axis, key in zip(axes[1, :3], METHODS):
        overlay = np.ma.masked_where(~strong_edge_mask, strong_edge_ratios[key])
        axis.imshow(images[key], cmap="gray", vmin=0.0, vmax=1.0)
        rendered = axis.imshow(
            overlay,
            cmap=colormap,
            vmin=RATIO_VMIN,
            vmax=RATIO_VMAX,
            alpha=0.9,
        )
        axis.set_axis_off()
        axis.set_title(f"{METHODS[key][0]} edge retention", fontsize=11, pad=8)

    axes[1, 3].imshow(strong_edge_mask, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 3].set_axis_off()
    axes[1, 3].set_title("GT strong-edge reference", fontsize=11, pad=8)
    scalar = plt.cm.ScalarMappable(
        cmap=colormap,
        norm=plt.Normalize(RATIO_VMIN, RATIO_VMAX),
    )
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=axes[1, :3].tolist(), fraction=0.025, pad=0.02)
    colorbar.set_label("edge strength: output / GT")
    figure.suptitle(
        f"BSD68 {sample_id}: edge preservation summary | blue=weaker, white=preserved, red=stronger",
        fontsize=13,
    )
    save_figure(figure, output / f"{sample_id}_edge_preservation_summary.png", dpi)


def render_edge_summary_individual(
    sample_id: str,
    images: OrderedDict,
    strong_edge_ratios: OrderedDict,
    strong_edge_mask: np.ndarray,
    scores: Dict[str, float],
    output: Path,
    dpi: int,
) -> None:
    colormap = plt.get_cmap("coolwarm").copy()
    colormap.set_bad((0.12, 0.12, 0.12, 1.0))
    for key in METHODS:
        figure, axis = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
        overlay = np.ma.masked_where(~strong_edge_mask, strong_edge_ratios[key])
        axis.imshow(images[key], cmap="gray", vmin=0.0, vmax=1.0)
        rendered = axis.imshow(
            overlay,
            cmap=colormap,
            vmin=RATIO_VMIN,
            vmax=RATIO_VMAX,
            alpha=0.9,
        )
        axis.set_axis_off()
        axis.set_title(
            f"BSD68 {sample_id}: {METHODS[key][0]} edge retention | {scores[key]:.2f} dB",
            fontsize=11,
            pad=8,
        )
        colorbar = figure.colorbar(rendered, ax=axis, shrink=0.8)
        colorbar.set_label("edge strength: output / GT")
        save_figure(figure, output / f"{sample_id}_{key}_edge_retention.png", dpi)

    figure, axis = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    axis.imshow(strong_edge_mask, cmap="gray", vmin=0.0, vmax=1.0)
    axis.set_axis_off()
    axis.set_title(f"BSD68 {sample_id}: GT strong-edge reference", fontsize=12, pad=8)
    save_figure(figure, output / f"{sample_id}_gt_strong_edge_reference.png", dpi)


def parse_samples(value: Optional[str]) -> Optional[set]:
    if not value:
        return None
    selected = {item.strip() for item in value.split(",") if item.strip()}
    return selected or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp8-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples", default=None)
    parser.add_argument("--error-percentile", type=float, default=99.5)
    parser.add_argument("--error-vmax", type=float, default=None)
    parser.add_argument("--edge-percentile", type=float, default=90.0)
    parser.add_argument("--edge-band-radius", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    roots = {
        "ours": args.exp8_root / "final",
        "ffdnet": args.official_root / "ffdnet_official_sigma25" / "coarse",
        "dncnn": args.official_root / "dncnn_official_sigma25" / "coarse",
        "gt": args.exp8_root / "gt",
    }
    files = {key: collect_files(path) for key, path in roots.items()}
    common_ids = set(files["gt"])
    for key in METHODS:
        common_ids &= set(files[key])
    selected = parse_samples(args.samples)
    if selected is not None:
        missing = selected - common_ids
        if missing:
            raise RuntimeError(f"Requested samples are missing from one or more outputs: {sorted(missing)}")
        sample_ids = sorted(selected)
    else:
        sample_ids = sorted(common_ids)
    if not sample_ids:
        raise RuntimeError("No common samples found across Exp8A, FFDNet, DnCNN, and GT outputs")

    loaded = OrderedDict()
    all_errors = []
    edge_data = OrderedDict()
    records = []
    for sample_id in sample_ids:
        images = OrderedDict((key, read_float(files[key][sample_id])) for key in METHODS)
        target = read_float(files["gt"][sample_id])
        for key, image in images.items():
            if image.shape != target.shape:
                raise ValueError(f"Shape mismatch for {sample_id}: {key}={image.shape}, gt={target.shape}")
            all_errors.append(np.abs(image - target))
        loaded[sample_id] = (images, target)

        gt_gx, gt_gy, gt_gradient = sobel_components(target)
        threshold = float(np.percentile(gt_gradient, args.edge_percentile))
        strong_edge_mask = gt_gradient >= threshold
        edge_band_mask = dilate_mask(strong_edge_mask, args.edge_band_radius)
        gradients = OrderedDict([("gt", gt_gradient)])
        gradient_errors = OrderedDict()
        strong_edge_ratios = OrderedDict()
        for key, image in images.items():
            gx, gy, gradient = sobel_components(image)
            gradients[key] = gradient
            gradient_errors[key] = np.sqrt((gx - gt_gx) ** 2 + (gy - gt_gy) ** 2)
            ratio = np.full_like(gradient, np.nan, dtype=np.float32)
            denominator = np.maximum(gt_gradient[strong_edge_mask], 1e-6)
            ratio[strong_edge_mask] = gradient[strong_edge_mask] / denominator
            strong_edge_ratios[key] = ratio
        edge_data[sample_id] = {
            "strong_edge_mask": strong_edge_mask,
            "edge_band_mask": edge_band_mask,
            "gradients": gradients,
            "gradient_errors": gradient_errors,
            "strong_edge_ratios": strong_edge_ratios,
        }

    error_vmax = args.error_vmax or shared_percentile(all_errors, args.error_percentile)
    args.output_root.mkdir(parents=True, exist_ok=True)
    four_panel_root = args.output_root / "four_panel"
    individual_root = args.output_root / "individual"
    heatmap_root = args.output_root / "error_heatmaps"
    edge_summary_root = args.output_root / "edge_analysis" / "summary"
    edge_summary_individual_root = edge_summary_root / "individual"

    for sample_id, (images, target) in loaded.items():
        display_images = OrderedDict(images)
        display_images["gt"] = target
        scores = {}
        mses = {}
        for key in METHODS:
            scores[key], mses[key] = psnr(images[key], target)
        data = edge_data[sample_id]
        render_four_panel(sample_id, display_images, scores, four_panel_root, args.dpi)
        render_single_images(sample_id, display_images, scores, individual_root, args.dpi)
        render_error_heatmap(sample_id, images, target, scores, error_vmax, heatmap_root, args.dpi)
        render_edge_summary(
            sample_id,
            display_images,
            data["strong_edge_ratios"],
            data["strong_edge_mask"],
            scores,
            edge_summary_root,
            args.dpi,
        )
        render_edge_summary_individual(
            sample_id,
            images,
            data["strong_edge_ratios"],
            data["strong_edge_mask"],
            scores,
            edge_summary_individual_root,
            args.dpi,
        )

        edge_metrics = {}
        for key in METHODS:
            band_error = data["gradient_errors"][key][data["edge_band_mask"]]
            ratio_values = data["strong_edge_ratios"][key][data["strong_edge_mask"]]
            edge_metrics[f"{key}_edge_band_gradient_error_mae"] = float(np.mean(band_error))
            edge_metrics[f"{key}_edge_band_gradient_error_p95"] = float(np.percentile(band_error, 95))
            edge_metrics[f"{key}_edge_gradient_ratio_mean"] = float(np.mean(ratio_values))
            edge_metrics[f"{key}_edge_gradient_ratio_median"] = float(np.median(ratio_values))
            edge_metrics[f"{key}_edge_gradient_retention_fraction"] = float(
                np.mean((ratio_values >= 0.85) & (ratio_values <= 1.15))
            )
            edge_metrics[f"{key}_edge_gradient_weak_fraction"] = float(np.mean(ratio_values < 0.85))
            edge_metrics[f"{key}_edge_gradient_strong_fraction"] = float(np.mean(ratio_values > 1.15))
        records.append(
            {
                "sample": sample_id,
                **{f"{key}_psnr": scores[key] for key in METHODS},
                **{f"{key}_mse": mses[key] for key in METHODS},
                **edge_metrics,
            }
        )

    with (args.output_root / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = (
        "Denoise visualization: Exp8A vs official FFDNet/DnCNN\n"
        f"Samples: {len(sample_ids)}\n"
        "Image display range: [0, 1] for every panel; no per-image min-max normalization\n"
        f"Error display range: [0, {error_vmax:.9g}] shared by every method and selected sample\n"
        f"GT strong-edge threshold: {args.edge_percentile:g}th percentile; band radius={args.edge_band_radius}\n"
        f"Edge ratio display range: [{RATIO_VMIN}, {RATIO_VMAX}], shared by every method; outside GT strong edges is masked\n"
        "Edge outputs: summary and summary/individual only\n"
        f"Source Exp8A: {args.exp8_root}\n"
        f"Source official models: {args.official_root}\n"
        f"Output: {args.output_root}\n"
    )
    (args.output_root / "normalization.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print(f"Saved four-panel figures: {four_panel_root}", flush=True)
    print(f"Saved individual figures: {individual_root}", flush=True)
    print(f"Saved error heatmaps: {heatmap_root}", flush=True)
    print(f"Saved edge summaries: {edge_summary_root}", flush=True)


if __name__ == "__main__":
    main()
