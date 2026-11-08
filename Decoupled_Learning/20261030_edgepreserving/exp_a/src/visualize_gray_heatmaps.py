"""Visualize grayscale prediction errors against the WLS test labels."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_A_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/mnt/d/ResearchCode/S2R-HDR/data/test_fei/generated_20260818")
DEFAULT_PREDICTION_ROOT = EXP_A_ROOT / "evaluation" / "test_fei_modelvis"


def read_unit_gray(path: Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 2:
        raise ValueError(f"Expected a readable grayscale image: {path}")
    image = raw.astype(np.float32)
    if np.issubdtype(raw.dtype, np.integer):
        image /= float(np.iinfo(raw.dtype).max)
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite pixels in {path}")
    return np.clip(image, 0.0, 1.0)


def gradient_xy(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Normalize the 3x3 Sobel derivative to approximate gray-level change per pixel.
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    return dx, dy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--vmax", type=float, default=None,
                        help="Fixed absolute-error color scale in normalized [0,1] units")
    args = parser.parse_args()

    tag = "wls_l0p500_a1p400"
    param_root = args.prediction_root / tag
    label_dir = args.data_root / "testlabel-l0p500-a1p400"
    output_dir = args.output_dir or param_root / "gray_heatmaps"
    model_dirs = {
        "NAFNet": param_root / "nafnet",
        "VIVNet": param_root / "vivnet",
        "Exp7C": param_root / "exp7c_recipe",
    }
    label_paths = sorted(label_dir.glob("*-WLS-l0p500-a1p400.tiff"))
    if not label_paths:
        raise FileNotFoundError(f"No WLS labels found in {label_dir}")

    samples = []
    all_errors = []
    all_edge_errors = []
    for label_path in label_paths:
        scene = label_path.name.removesuffix("-WLS-l0p500-a1p400.tiff")
        target = read_unit_gray(label_path)
        predictions = {}
        for model, directory in model_dirs.items():
            pred_path = directory / f"{scene}.tiff"
            prediction = read_unit_gray(pred_path)
            predictions[model] = prediction

        # Exp7C uses even-sized tensors, so odd-sized samples lose the last
        # row and column. Compare every model on that same shared region.
        height = min([target.shape[0], *(image.shape[0] for image in predictions.values())])
        width = min([target.shape[1], *(image.shape[1] for image in predictions.values())])
        target = target[:height, :width]
        predictions = {
            model: image[:height, :width] for model, image in predictions.items()
        }
        errors = {}
        edge_errors = {}
        target_dx, target_dy = gradient_xy(target)
        for model, prediction in predictions.items():
            errors[model] = np.abs(prediction - target)
            all_errors.append(errors[model].ravel())
            pred_dx, pred_dy = gradient_xy(prediction)
            edge_errors[model] = np.sqrt(
                np.square(pred_dx - target_dx) + np.square(pred_dy - target_dy)
            )
            all_edge_errors.append(edge_errors[model].ravel())
        samples.append((scene, target, predictions, errors, edge_errors, target_dx, target_dy))

    vmax = args.vmax
    if vmax is None:
        vmax = float(np.percentile(np.concatenate(all_errors), 99.5))
    vmax = max(vmax, 1e-8)
    edge_vmax = float(np.percentile(np.concatenate(all_edge_errors), 99.5))
    edge_vmax = max(edge_vmax, 1e-8)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "gray_error_metrics.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["scene", "model", "mae", "rmse", "max_abs_error", "error_p99",
                  "edge_mae", "edge_rmse", "edge_max_error", "edge_p99"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scene, target, predictions, errors, edge_errors, target_dx, target_dy in samples:
            figure, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
            axes[0].imshow(target, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("WLS target")
            for axis, (model, prediction) in zip(axes[1:], predictions.items()):
                error = errors[model]
                axis.imshow(error, cmap="inferno", vmin=0, vmax=vmax)
                axis.set_title(f"{model} |pred - WLS|")
                writer.writerow({
                    "scene": scene,
                    "model": model,
                    "mae": float(error.mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "max_abs_error": float(error.max()),
                    "error_p99": float(np.percentile(error, 99)),
                    "edge_mae": float(edge_errors[model].mean()),
                    "edge_rmse": float(np.sqrt(np.mean(np.square(edge_errors[model])))),
                    "edge_max_error": float(edge_errors[model].max()),
                    "edge_p99": float(np.percentile(edge_errors[model], 99)),
                })
                fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
                image = ax.imshow(error, cmap="inferno", vmin=0, vmax=vmax)
                ax.set_title(f"{scene} - {model} absolute grayscale error")
                ax.axis("off")
                fig.colorbar(image, ax=ax, label="Absolute error (normalized [0,1])")
                fig.savefig(output_dir / f"{scene}_{model.lower()}_error.png", dpi=160)
                plt.close(fig)
            for axis in axes:
                axis.axis("off")
            figure.colorbar(axes[-1].images[0], ax=axes[1:],
                            label="Absolute error (normalized [0,1])", shrink=0.82)
            figure.suptitle(f"{scene} | shared error scale: 0 to {vmax:.5g}")
            figure.savefig(output_dir / f"{scene}_gray_error_compare.png", dpi=160)
            plt.close(figure)

            edge_figure, edge_axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
            target_edge_magnitude = np.sqrt(np.square(target_dx) + np.square(target_dy))
            edge_axes[0].imshow(target_edge_magnitude, cmap="gray", vmin=0)
            edge_axes[0].set_title("WLS target gradient magnitude")
            for axis, model in zip(edge_axes[1:], predictions):
                edge_image = axis.imshow(edge_errors[model], cmap="inferno", vmin=0, vmax=edge_vmax)
                axis.set_title(f"{model} gradient error")
            for axis in edge_axes:
                axis.axis("off")
            edge_figure.colorbar(edge_image, ax=edge_axes[1:],
                                 label="Gradient-vector error (gray level / pixel)", shrink=0.82)
            edge_figure.suptitle(
                f"{scene} | shared edge-error scale: 0 to {edge_vmax:.5g}"
            )
            edge_figure.savefig(output_dir / f"{scene}_edge_error_compare.png", dpi=160)
            plt.close(edge_figure)

    vmax_source = (
        "explicit" if args.vmax is not None
        else "global_99.5th_percentile_across_all_models_and_scenes"
    )
    (output_dir / "color_scale.txt").write_text(
        f"absolute_error_vmin=0\nabsolute_error_vmax={vmax:.10g}\n"
        f"vmax_source={vmax_source}\n",
        encoding="utf-8",
    )
    (output_dir / "edge_color_scale.txt").write_text(
        "edge_error=euclidean_difference_between_predicted_and_target_Sobel_gradient_vectors\n"
        "sobel=3x3_scale_0.125\n"
        f"edge_error_vmin=0\nedge_error_vmax={edge_vmax:.10g}\n"
        "vmax_source=global_99.5th_percentile_across_all_models_and_scenes\n",
        encoding="utf-8",
    )
    print(f"Saved {len(samples)} grayscale error comparisons to {output_dir}")


if __name__ == "__main__":
    main()
