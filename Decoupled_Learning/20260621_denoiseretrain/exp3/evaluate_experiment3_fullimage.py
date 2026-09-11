from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import FullImageGrayDenoisingDataset, list_image_paths
from model import DeepIndependentResidual3Stage
from train_experiment3_fullimage import calc_psnr, normalize_path_for_runtime


DEFAULT_DATA_ROOT = Path(r"D:\nuoqi\halo-free\data_bsds\BSD68")
DEFAULT_EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DEFAULT_EXPERIMENT_ROOT.parent / "exp1"))
from eval_convguided_gray import DeepGuidedFilter, load_model_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Experiment 3 on full BSD68 images.")
    parser.add_argument("--sigma", type=float, required=True, choices=(25.0,))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--stage1-model-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(normalize_path_for_runtime(args.data_root))
    experiment_root = Path(normalize_path_for_runtime(args.experiment_root))
    checkpoint = Path(normalize_path_for_runtime(args.checkpoint)) if args.checkpoint else experiment_root / "models" / f"sigma{args.sigma:g}" / f"experiment3_deep_sigma{args.sigma:g}_val_best.pth"
    output_dir = Path(normalize_path_for_runtime(args.output_dir)) if args.output_dir else experiment_root / "eval_results" / f"sigma{args.sigma:g}"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = FullImageGrayDenoisingDataset(list_image_paths(root), args.sigma, augment=False, deterministic_noise_seed=0)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    stage1_path = args.stage1_model_path or str(
        experiment_root.parent / "exp1" / "models" / f"convguided_bsds_sigma{args.sigma:g}" / f"convguided_bsds_sigma{args.sigma:g}_best.pth"
    )
    stage1_model = DeepGuidedFilter().to(device)
    load_model_weights(stage1_model, normalize_path_for_runtime(stage1_path), device)
    stage1_model.eval()
    model = DeepIndependentResidual3Stage().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    total_mse = total_psnr = total_noisy_psnr = 0.0
    with torch.no_grad():
        for batch in loader:
            noisy_lr = batch["noisy_lr"].to(device)
            noisy = batch["noisy_hr"].to(device)
            clean = batch["clean_hr"].to(device)
            coarse = stage1_model(noisy_lr - 0.5, noisy - 0.5)
            prediction = torch.clamp(model(noisy - 0.5, coarse - 0.5)["final"] + 0.5, 0.0, 1.0)
            psnr, mse = calc_psnr(prediction, clean)
            noisy_psnr, _ = calc_psnr(noisy, clean)
            total_mse += mse
            total_psnr += psnr
            total_noisy_psnr += noisy_psnr
            image = np.rint(prediction[0, 0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / str(batch["filename"][0])), image)
    count = len(dataset)
    metrics = {"sigma": args.sigma, "images": count, "avg_mse": total_mse / count, "avg_psnr": total_psnr / count, "avg_noisy_psnr": total_noisy_psnr / count, "gain_over_noisy": (total_psnr - total_noisy_psnr) / count, "checkpoint": str(checkpoint)}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
