from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


PRED_A_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/test_vivo/rf3stage_otsu_output_v5")
PRED_B_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/test_vivo/df_output")
OUTPUT_DIR = Path(
    "/mnt/d/nuoqi/halo-free/Decoupled_Learning/DeepGuidedFilter/metric_cal/deltapred_vis"
)

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
VIS_PERCENTILE = 99.0


def collect_images(folder: Path):
    files = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files[path.stem] = path
    return files


def normalize_pair_key(stem: str):
    if stem.endswith("_pred_pred"):
        return stem[:-10]
    if stem.endswith("_pred"):
        return stem[:-5]
    return stem


def read_image(path: Path):
    img = imageio.imread(str(path))
    return img.astype(np.float32)


def save_float32_tiff(path: Path, img: np.ndarray):
    imageio.imwrite(str(path), img.astype(np.float32))


def crop_pair_to_common_shape(a: np.ndarray, b: np.ndarray):
    if a.ndim != b.ndim:
        raise ValueError(f"Dimension mismatch: {a.ndim} vs {b.ndim}")

    common_shape = tuple(min(x, y) for x, y in zip(a.shape, b.shape))
    slices = tuple(slice(0, size) for size in common_shape)
    return a[slices], b[slices]


def to_signed_vis(delta: np.ndarray, percentile: float = VIS_PERCENTILE):
    if delta.ndim == 3:
        delta_vis = delta.mean(axis=2)
    else:
        delta_vis = delta

    abs_max = float(np.percentile(np.abs(delta_vis), percentile))
    if abs_max < 1e-12:
        abs_max = 1.0

    normalized = np.clip(delta_vis / abs_max, -1.0, 1.0)
    colored = plt.get_cmap("seismic")((normalized + 1.0) * 0.5)[..., :3]
    return (colored * 255.0).astype(np.uint8), abs_max


def main():
    if not PRED_A_DIR.is_dir():
        raise FileNotFoundError(f"PRED_A_DIR not found: {PRED_A_DIR}")
    if not PRED_B_DIR.is_dir():
        raise FileNotFoundError(f"PRED_B_DIR not found: {PRED_B_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_a_files = collect_images(PRED_A_DIR)
    pred_b_files = collect_images(PRED_B_DIR)
    pred_a_map = {normalize_pair_key(name): path for name, path in pred_a_files.items()}
    pred_b_map = {normalize_pair_key(name): path for name, path in pred_b_files.items()}
    common_names = sorted(set(pred_a_map) & set(pred_b_map))

    if not common_names:
        raise RuntimeError("No matched filenames found between the two prediction folders")

    print(f"[Info] Matched {len(common_names)} image pairs.")
    print(f"[Info] Output dir: {OUTPUT_DIR}")

    for idx, name in enumerate(common_names, start=1):
        pred_a = read_image(pred_a_map[name])
        pred_b = read_image(pred_b_map[name])
        pred_a, pred_b = crop_pair_to_common_shape(pred_a, pred_b)

        delta = pred_a - pred_b
        delta_tiff_path = OUTPUT_DIR / f"{name}_minus_df.tiff"
        delta_png_path = OUTPUT_DIR / f"{name}_minus_df_signed_vis.png"

        save_float32_tiff(delta_tiff_path, delta)
        delta_vis, vis_abs_max = to_signed_vis(delta)
        imageio.imwrite(delta_png_path, delta_vis)

        print(
            f"[{idx}/{len(common_names)}] {name} | "
            f"shape={delta.shape} | min={delta.min():.6f} | max={delta.max():.6f} | "
            f"vis_range=[{-vis_abs_max:.6f}, {vis_abs_max:.6f}]"
        )

    print("[Done] Delta prediction TIFFs and signed visualization PNGs saved.")


if __name__ == "__main__":
    main()
