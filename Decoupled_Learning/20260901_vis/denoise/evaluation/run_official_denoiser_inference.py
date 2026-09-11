"""Evaluate the official DnCNN and FFDNet checkpoints on BSD68."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


HERE = Path(__file__).resolve().parents[1]
TRAIN_ROOT = HERE.parents[1] / "20260621_denoiseretrain"
DEFAULT_DNCNN = TRAIN_ROOT / "third_party/DnCNN/TrainingCodes/dncnn_pytorch/models/DnCNN_sigma25/model.pth"
DEFAULT_FFDNET = TRAIN_ROOT / "third_party/FFDNet/refine_bridge/pytorch_official/models/ffdnet_gray.pth"
SIGMA = 25.0

sys.path.insert(0, str(TRAIN_ROOT))
from exp7_8_common import load_denoiser  # noqa: E402
from run_exp8a_inference import crop_even, image_files, read_gray, save_float  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--dncnn-checkpoint", type=Path, default=DEFAULT_DNCNN)
    parser.add_argument("--ffdnet-checkpoint", type=Path, default=DEFAULT_FFDNET)
    args = parser.parse_args()

    samples = image_files(args.data_root)
    if not samples:
        raise RuntimeError(f"No image samples found in BSD68 directory: {args.data_root}")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    models = {
        "dncnn_official_sigma25": load_denoiser("dncnn", str(args.dncnn_checkpoint), device),
        "ffdnet_official_sigma25": load_denoiser("ffdnet", str(args.ffdnet_checkpoint), device),
    }
    for model in models.values():
        model.eval()

    metrics = {name: [] for name in models}
    for name in models:
        (args.output_root / name / "source").mkdir(parents=True, exist_ok=True)
        (args.output_root / name / "noisy").mkdir(parents=True, exist_ok=True)
        (args.output_root / name / "coarse").mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for sample_path in tqdm(samples, desc="Official DnCNN + FFDNet", unit="image"):
            sample_id = sample_path.stem
            clean_np = crop_even(read_gray(sample_path))
            rng = np.random.RandomState(args.noise_seed)
            noisy_np = clean_np + rng.normal(0.0, SIGMA / 255.0, clean_np.shape).astype("float32")
            noisy = torch.from_numpy(noisy_np)[None, None].to(device)
            for name, model in models.items():
                kind = "dncnn" if name.startswith("dncnn") else "ffdnet"
                coarse = load_denoiser_forward(model, noisy, kind).clamp(0, 1)
                coarse_np = coarse[0, 0].cpu().numpy()
                output_dir = args.output_root / name
                save_float(output_dir / "source" / f"{sample_id}.tiff", clean_np)
                save_float(output_dir / "noisy" / f"{sample_id}.tiff", noisy_np)
                save_float(output_dir / "coarse" / f"{sample_id}.tiff", coarse_np)
                mse = float(np.mean((coarse_np - clean_np) ** 2))
                metrics[name].append(100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse))

    report = [
        "Official denoiser BSD68 inference",
        "Data: direct BSD68 clean images",
        "Noise: online sigma25 AWGN, RandomState(0) per image, no noisy clamp",
        f"Samples: {len(samples)}",
        "",
    ]
    for name, scores in metrics.items():
        report.extend([
            f"Model: {name}",
            f"Checkpoint: {args.dncnn_checkpoint if name.startswith('dncnn') else args.ffdnet_checkpoint}",
            f"Mean per-image PSNR: {sum(scores) / len(scores):.4f} dB",
            "",
        ])
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "metrics.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


def load_denoiser_forward(model, noisy, kind):
    if kind == "ffdnet":
        sigma = noisy.new_full((noisy.shape[0], 1, 1, 1), SIGMA / 255.0)
        return model(noisy, sigma)
    return model(noisy)


if __name__ == "__main__":
    main()
