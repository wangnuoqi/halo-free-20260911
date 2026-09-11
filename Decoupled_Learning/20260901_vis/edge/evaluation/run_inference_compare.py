"""Run baseline and exp7C-recipe pipelines on one matched test set.

Outputs are normalized float32 TIFFs in [0, 1].  The MATLAB script consumes
these files as filtered grayscale base images and reconstructs display RGB.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import re
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm


HERE = Path(__file__).resolve().parents[1]
MODEL_ROOT = HERE / "models"


def load_definitions(path: Path, name: str):
    source = path.read_text(encoding="utf-8")
    # Training files have a bootstrap below this point.  Keep only imports,
    # model classes, and helper functions when importing them for inference.
    markers = [
        "\n###### 训练输入",  # original baseline starts its dataset bootstrap here
        "\nif os.name == \"nt\":",  # exp7C/exp8 copies start path setup here
        "\ndevice = torch.device",
        "\nif __name__ == \"__main__\":",
    ]
    cut = len(source)
    for marker in markers:
        pos = source.find(marker)
        if pos >= 0:
            cut = min(cut, pos)
    namespace = {"__name__": name, "__file__": str(path)}
    exec(compile(source[:cut], str(path), "exec"), namespace)
    return type(name, (), namespace)


def load_state(model, path: Path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def numeric_id(path: Path):
    match = re.match(r"(\d+)", path.name)
    return match.group(1) if match else None


def sample_key(path: Path):
    """Return the scene id shared by input, LR, and WLS-label filenames."""
    stem = path.stem
    stem = re.sub(r"-WLS(?:-|$).*", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"-log$", "", stem, flags=re.IGNORECASE)
    return stem or numeric_id(path)


def index_files(path: Path):
    return {sample_key(p): p for p in path.iterdir() if p.is_file() and sample_key(p)}


def resolve_test_dirs(data_root: Path, parameter_tag: str | None = None):
    """Resolve both flat and parameter-suffixed test directory layouts."""
    if parameter_tag:
        suffix = parameter_tag if parameter_tag.startswith("l") else f"{parameter_tag}"
        parameter_origin = data_root / f"testinput-{suffix}"
        parameter_label = data_root / f"testlabel-{suffix}"
        parameter_lr = data_root / f"testinput-{suffix}_downsample"
        if parameter_origin.is_dir() and parameter_label.is_dir():
            return parameter_origin, parameter_label, parameter_lr
        raise RuntimeError(f"Cannot resolve parameterized test directories for {suffix} under {data_root}")

    direct_origin = data_root / "testinput"
    direct_label = data_root / "testlabel"
    if direct_origin.is_dir() and direct_label.is_dir():
        for direct_lr in (data_root / "testinput_downsample",
                          data_root / "down_sample_2_testinput"):
            if direct_lr.is_dir():
                return direct_origin, direct_label, direct_lr
        return direct_origin, direct_label, data_root / "testinput_downsample"

    inputs = sorted(path for path in data_root.glob("testinput-*")
                    if path.is_dir() and not path.name.endswith("_downsample"))
    candidates = []
    for origin in inputs:
        suffix = origin.name[len("testinput-"):]
        label = data_root / f"testlabel-{suffix}"
        lr = data_root / f"testinput-{suffix}_downsample"
        if label.is_dir():
            # Prefer the exact test subset that also has precomputed LR.
            candidates.append((lr.is_dir(), origin, label, lr))
    if not candidates:
        raise RuntimeError(f"Cannot resolve testinput/testlabel directories under {data_root}")
    _, origin, label, lr = max(candidates, key=lambda item: item[0])
    return origin, label, lr


def resolve_model_config(data_root: Path, label_dir: Path | None = None):
    """Select models trained for the requested label set."""
    # test_fei generated sets keep the training label protocol in their
    # directory name, while the flat test_fei set uses the legacy mixed model.
    root_name = data_root.name.lower()
    label_name = label_dir.name.lower() if label_dir is not None else ""
    if root_name == "test_vivo" and "l1p000-a1p600" in label_name:
        baseline = MODEL_ROOT / "baseline_20260813"
        improved = MODEL_ROOT / "exp7c_recipe_20260813"
    elif root_name == "test_vivo" and "l0p500-a1p400" in label_name:
        baseline = MODEL_ROOT / "baseline_20260818"
        improved = MODEL_ROOT / "exp7c_recipe_20260818"
    elif root_name == "test_vivo" and "l0p250-a1p200" in label_name:
        baseline = MODEL_ROOT / "baseline_original"
        improved = MODEL_ROOT / "exp7c_recipe_mixed"
    elif root_name in {"20260813", "generated_20260813"}:
        baseline = MODEL_ROOT / "baseline_20260813"
        improved = MODEL_ROOT / "exp7c_recipe_20260813"
    elif root_name in {"20260818", "generated_20260818"}:
        baseline = MODEL_ROOT / "baseline_20260818"
        improved = MODEL_ROOT / "exp7c_recipe_20260818"
    elif root_name in {"mixed_lavel_singlehdr", "testinput", "test_fei"}:
        baseline = MODEL_ROOT / "baseline_original"
        improved = MODEL_ROOT / "exp7c_recipe_mixed"
    else:
        raise ValueError(
            f"No model mapping for {data_root}. Supported roots: 20260813, 20260818, generated_20260813, generated_20260818, mixed_lavel_singlehdr, test_fei, testinput")
    return {
        "baseline_conv": (baseline / "convguided_filter_best_mix.pth"
                          if baseline.name == "baseline_original" else baseline / "convguided.pth"),
        "baseline_rf": (baseline / "rf_3stage_best_otsu_v3.pth"
                        if baseline.name == "baseline_original" else baseline / "rf.pth"),
        "new_conv": (improved / "convguided_joint_best_20260818_l0p500_a1p400.pth"
                     if improved.name == "exp7c_recipe_mixed" else improved / "convguided.pth"),
        "new_rf": (improved / "rf_3stage_best_20260818_l0p500_a1p400.pth"
                   if improved.name == "exp7c_recipe_mixed" else improved / "rf.pth"),
        "baseline_dir": baseline,
        "new_dir": improved,
    }


def crop_even(array):
    h, w = array.shape[-2:]
    return array[..., : h - h % 2, : w - w % 2]


def downsample_half_long_edge(array):
    from PIL import Image
    h, w = array.shape
    h -= h % 2
    w -= w % 2
    array = array[:h, :w]
    if h >= w:
        new_h = max(1, h // 2)
        new_w = max(1, round(w * new_h / h))
    else:
        new_w = max(1, w // 2)
        new_h = max(1, round(h * new_w / w))
    return np.asarray(Image.fromarray(array.astype("float32"), mode="F").resize(
        (new_w, new_h), resample=Image.Resampling.LANCZOS), dtype="float32")


def read_gray(path: Path):
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise RuntimeError(f"Cannot read {path}")
    return value.astype("float32")


def save_float(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, value.astype("float32"))


def summarize_metrics(mse_values):
    mean_mse = sum(mse_values) / len(mse_values)
    per_image_psnr = [100.0 if mse == 0 else 10 * math.log10(1 / mse)
                      for mse in mse_values]
    return {
        "mean_mse": mean_mse,
        "psnr_from_mean_mse": 100.0 if mean_mse == 0 else 10 * math.log10(1 / mean_mse),
        "mean_per_image_psnr": sum(per_image_psnr) / len(per_image_psnr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path,
                    default=Path("/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr"))
    ap.add_argument("--output-root", type=Path, default=HERE / "outputs" / "mixed_lavel_singlehdr")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--parameter-tag", default=None,
                    help="Select e.g. l1p000-a1p600 under a flat dataset root")
    args = ap.parse_args()

    data = args.data_root
    parameter_tag = args.parameter_tag
    if parameter_tag is None:
        match = re.search(r"wls_(l[0-9]+p[0-9]+-a[0-9]+p[0-9]+)", str(args.output_root))
        parameter_tag = match.group(1) if match else None
    origin_dir, label_dir, lr_dir = resolve_test_dirs(data, parameter_tag)
    lr_files = index_files(lr_dir) if lr_dir.is_dir() else {}
    origin_files = index_files(origin_dir)
    label_files = index_files(label_dir)
    ids = sorted(set(origin_files) & set(label_files))
    if not ids:
        raise RuntimeError(f"No matched testinput/testlabel files under {data}")
    print(f"Input: {origin_dir}")
    print(f"Label: {label_dir}")
    print(f"LR: {lr_dir if lr_dir.is_dir() else 'generated online when needed'}")
    print(f"Matched test samples: {len(ids)}")

    config = resolve_model_config(data, label_dir)
    baseline_dir = config["baseline_dir"]
    new_dir = config["new_dir"]
    baseline_rf = load_definitions(baseline_dir / "refine_definition.py", "baseline_rf")
    new_rf = load_definitions(new_dir / "exp7c_recipe_definition.py", "exp7c_rf")
    conv_def = load_definitions(new_dir / "exp7c_recipe_definition.py", "conv_def")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    baseline_conv = conv_def.DeepGuidedFilter(radius=1, eps=1e-8).to(device)
    new_conv = conv_def.DeepGuidedFilter(radius=1, eps=1e-8).to(device)
    baseline_model = baseline_rf.CascadeRefine3Stage(
        base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True).to(device)
    new_model = new_rf.CascadeRefine3Stage(
        base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True).to(device)
    load_state(baseline_conv, config["baseline_conv"])
    load_state(new_conv, config["new_conv"])
    load_state(baseline_model, config["baseline_rf"])
    load_state(new_model, config["new_rf"])
    for model in (baseline_conv, new_conv, baseline_model, new_model):
        model.eval()

    out_base = args.output_root / "baseline_original"
    out_new = args.output_root / "exp7c_recipe"
    out_gt = args.output_root / "gt"
    mse_base, mse_new = [], []
    with torch.inference_mode():
        for sample_id in tqdm(ids, desc="Inference", unit="image"):
            origin_np = read_gray(origin_files[sample_id])
            label_np = read_gray(label_files[sample_id])
            lr_np = read_gray(lr_files[sample_id]) if sample_id in lr_files else downsample_half_long_edge(origin_np)
            origin = torch.from_numpy(crop_even(origin_np))[None, None].to(device)
            label = torch.from_numpy(crop_even(label_np))[None, None].to(device)
            lr = torch.from_numpy(lr_np)[None, None].to(device)
            base_coarse = baseline_conv(lr - 0.5, origin - 0.5)
            new_coarse = new_conv(lr - 0.5, origin - 0.5)
            base_pred = torch.clamp(baseline_model(origin - 0.5, base_coarse - 0.5)["final"] + 0.5, 0, 1)
            new_pred = torch.clamp(new_model(origin - 0.5, new_coarse - 0.5)["final"] + 0.5, 0, 1)
            base = base_pred[0, 0].cpu().numpy()
            new = new_pred[0, 0].cpu().numpy()
            gt = label[0, 0].cpu().numpy()
            mse_base.append(float(np.mean((base - gt) ** 2)))
            mse_new.append(float(np.mean((new - gt) ** 2)))
            filename = f"{sample_id}.tiff"
            save_float(out_base / filename, base)
            save_float(out_new / filename, new)
            save_float(out_gt / filename, gt)

    base_metrics = summarize_metrics(mse_base)
    new_metrics = summarize_metrics(mse_new)
    report = (f"Samples: {len(ids)}\n"
              f"Baseline mean MSE: {base_metrics['mean_mse']:.8f}\n"
              f"Baseline PSNR from mean MSE: {base_metrics['psnr_from_mean_mse']:.4f} dB\n"
              f"Baseline mean per-image PSNR (training-test protocol): {base_metrics['mean_per_image_psnr']:.4f} dB\n"
              f"Exp7C recipe mean MSE: {new_metrics['mean_mse']:.8f}\n"
              f"Exp7C recipe PSNR from mean MSE: {new_metrics['psnr_from_mean_mse']:.4f} dB\n"
              f"Exp7C recipe mean per-image PSNR (training-test protocol): {new_metrics['mean_per_image_psnr']:.4f} dB\n")
    (args.output_root / "metrics.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
