"""Run the Exp8A patch-trained ConvGuided + frozen Exp6D RF pipeline."""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm


HERE = Path(__file__).resolve().parents[1]
MODEL_DIR = HERE / "models" / "exp8A_patch_frozen_sigma25"
TRAIN_ROOT = HERE.parents[1] / "20260621_denoiseretrain"
SIGMA = 25.0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_state(model: torch.nn.Module, path: Path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict):
        state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def image_files(path: Path):
    extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(item for item in path.iterdir()
                  if item.is_file() and item.suffix.lower() in extensions)


def crop_even(value: np.ndarray) -> np.ndarray:
    height, width = value.shape[-2:]
    return value[..., : height - height % 2, : width - width % 2]


def read_gray(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise RuntimeError(f"Cannot read {path}")
    if value.ndim == 3:
        value = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
    value = value.astype("float32")
    if value.max(initial=0.0) > 1.0:
        value /= 255.0
    return value


def save_float(path: Path, value: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, value.astype("float32"))


def add_official_test_noise(clean: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, sigma / 255.0, clean.shape).astype("float32")
    return clean + noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--noise-seed", type=int, default=0)
    args = parser.parse_args()

    samples = image_files(args.data_root)
    if not samples:
        raise RuntimeError(f"No image samples found in BSD68 directory: {args.data_root}")

    conv_def = load_module(MODEL_DIR / "convguided_definition.py", "exp8a_convguided")
    common = load_module(TRAIN_ROOT / "exp7_8_common.py", "exp8a_common")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    conv = conv_def.ConvGuidedGray().to(device)
    refine = common.Exp6Refine("exp6d").to(device)
    load_state(conv, MODEL_DIR / "convguided_patch_sigma25_best.pth")
    load_state(refine, MODEL_DIR / "refine_patch_exp6d_sigma25_best.pth")
    conv.eval()
    refine.eval()

    noisy_dir = args.output_root / "noisy"
    source_dir = args.output_root / "source"
    coarse_dir = args.output_root / "coarse"
    final_dir = args.output_root / "final"
    gt_dir = args.output_root / "gt"
    mse_coarse, mse_final = [], []
    psnr_coarse, psnr_final = [], []
    with torch.inference_mode():
        for sample_path in tqdm(samples, desc="Exp8A inference", unit="image"):
            sample_id = sample_path.stem
            clean_np = crop_even(read_gray(sample_path))
            label_np = clean_np
            noisy_np = add_official_test_noise(clean_np, SIGMA, args.noise_seed)
            noisy = torch.from_numpy(noisy_np)[None, None].to(device)
            lr = common.lanczos_downsample_half(noisy)
            coarse = conv(lr, noisy).clamp(0, 1)
            final = refine(noisy, coarse)["final"].clamp(0, 1)
            coarse_np = coarse[0, 0].cpu().numpy()
            final_np = final[0, 0].cpu().numpy()
            save_float(source_dir / f"{sample_id}.tiff", clean_np)
            save_float(noisy_dir / f"{sample_id}.tiff", noisy_np)
            save_float(coarse_dir / f"{sample_id}.tiff", coarse_np)
            save_float(final_dir / f"{sample_id}.tiff", final_np)
            save_float(gt_dir / f"{sample_id}.tiff", label_np)
            coarse_mse = float(np.mean((coarse_np - label_np) ** 2))
            final_mse = float(np.mean((final_np - label_np) ** 2))
            mse_coarse.append(coarse_mse)
            mse_final.append(final_mse)
            psnr_coarse.append(100.0 if coarse_mse == 0 else 10.0 * math.log10(1.0 / coarse_mse))
            psnr_final.append(100.0 if final_mse == 0 else 10.0 * math.log10(1.0 / final_mse))

    def psnr(values):
        mean_mse = sum(values) / len(values)
        return 100.0 if mean_mse == 0 else 10.0 * math.log10(1.0 / mean_mse)

    report = (
        f"Model: Exp8A ConvGuided patch base + frozen Exp6D RF\n"
        f"Noise: online sigma25 AWGN, RandomState({args.noise_seed}) per image, no noisy clamp\n"
        f"LR: Lanczos-4 half downsample from noisy HR\n"
        f"Data: direct BSD68 clean images used as input labels\n"
        f"Samples: {len(samples)}\n"
        f"Coarse mean MSE: {sum(mse_coarse) / len(mse_coarse):.8f}\n"
        f"Coarse mean per-image PSNR: {sum(psnr_coarse) / len(psnr_coarse):.4f} dB\n"
        f"Coarse PSNR from mean MSE (reference): {psnr(mse_coarse):.4f} dB\n"
        f"Final mean MSE: {sum(mse_final) / len(mse_final):.8f}\n"
        f"Final mean per-image PSNR: {sum(psnr_final) / len(psnr_final):.4f} dB\n"
        f"Final PSNR from mean MSE (reference): {psnr(mse_final):.4f} dB\n"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "metrics.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
