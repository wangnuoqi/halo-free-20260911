"""Experiment 2: independent residual refinement after frozen ConvGuided."""

from __future__ import annotations

import argparse
import datetime
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import build_datasets
from loss import staged_image_loss
from model import IndependentResidual3Stage


DEFAULT_DATA_ROOT = Path(r"D:\nuoqi\halo-free\data_bsds")
DEFAULT_EXPERIMENT_ROOT = Path(__file__).resolve().parent
EXP1_ROOT = DEFAULT_EXPERIMENT_ROOT.parent / "exp1"
sys.path.insert(0, str(EXP1_ROOT))
from eval_convguided_gray import DeepGuidedFilter, load_model_weights


def normalize_path_for_runtime(path: str) -> str:
    if not path or path[1:3] != ":\\":
        return path
    suffix = path[3:].replace("\\", "/")
    return f"/mnt/{path[0].lower()}/{suffix}"


def calc_psnr(prediction: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    mse = torch.mean((prediction - target).square()).item()
    return (100.0 if mse == 0.0 else 10.0 * np.log10(1.0 / mse), mse)


def save_model(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint: {path}")


@torch.no_grad()
def evaluate(
    model: IndependentResidual3Stage,
    stage1_model: torch.nn.Module,
    loader: Iterable[Dict[str, object]],
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_mse = 0.0
    total_psnr = 0.0
    total_noisy_psnr = 0.0
    total_images = 0
    for batch in loader:
        noisy_lr = batch["noisy_lr"].to(device, non_blocking=True)
        noisy = batch["noisy_hr"].to(device, non_blocking=True)
        clean = batch["clean_hr"].to(device, non_blocking=True)
        coarse = stage1_model(noisy_lr - 0.5, noisy - 0.5)
        prediction = torch.clamp(model(noisy - 0.5, coarse - 0.5)["final"] + 0.5, 0.0, 1.0)
        psnr, mse = calc_psnr(prediction, clean)
        noisy_psnr, _ = calc_psnr(noisy, clean)
        count = clean.shape[0]
        total_mse += mse * count
        total_psnr += psnr * count
        total_noisy_psnr += noisy_psnr * count
        total_images += count
    return total_mse / total_images, total_psnr / total_images, total_noisy_psnr / total_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 2: independent residual full-image refinement on BSDS400/BSD68."
    )
    parser.add_argument("--sigma", type=float, required=True, choices=(15.0, 25.0, 50.0))
    parser.add_argument("--train-data-root", default=str(DEFAULT_DATA_ROOT / "BSDS 400"))
    parser.add_argument("--validation-data-root", default=str(DEFAULT_DATA_ROOT / "BSD68"))
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--stage1-model-path", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Full BSDS images have mixed shapes; use --batch-size 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dataset, validation_dataset = build_datasets(
        normalize_path_for_runtime(args.train_data_root),
        normalize_path_for_runtime(args.validation_data_root),
        sigma=args.sigma,
    )
    print(f"Experiment 2 datasets: train={len(train_dataset)}, BSD68 validation={len(validation_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    stage1_path = args.stage1_model_path or str(
        EXP1_ROOT / "models" / f"convguided_bsds_sigma{args.sigma:g}" / f"convguided_bsds_sigma{args.sigma:g}_best.pth"
    )
    stage1_model = DeepGuidedFilter().to(device)
    load_model_weights(stage1_model, normalize_path_for_runtime(stage1_path), device)
    stage1_model.eval()
    for parameter in stage1_model.parameters():
        parameter.requires_grad_(False)
    model = IndependentResidual3Stage().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    output_dir = Path(normalize_path_for_runtime(args.experiment_root)) / "models" / f"sigma{args.sigma:g}"
    train_best_path = output_dir / f"experiment2_sigma{args.sigma:g}_train_best.pth"
    validation_best_path = output_dir / f"experiment2_sigma{args.sigma:g}_val_best.pth"
    validation_psnr_best_path = output_dir / f"experiment2_sigma{args.sigma:g}_val_psnr_best.pth"
    log_dir = Path(normalize_path_for_runtime(args.experiment_root)) / "log_dir" / f"experiment2_sigma{args.sigma:g}_{datetime.datetime.now():%Y%m%d-%H%M%S}"
    writer = SummaryWriter(log_dir=str(log_dir))
    best_train_mse = float("inf")
    best_validation_mse = float("inf")
    best_validation_psnr = float("-inf")
    try:
        for epoch in range(args.epochs):
            model.train()
            total_mse = total_psnr = 0.0
            total_images = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
                noisy_lr = batch["noisy_lr"].to(device, non_blocking=True)
                noisy = batch["noisy_hr"].to(device, non_blocking=True)
                clean = batch["clean_hr"].to(device, non_blocking=True)
                with torch.no_grad():
                    coarse = stage1_model(noisy_lr - 0.5, noisy - 0.5)
                outputs = model(noisy - 0.5, coarse - 0.5)
                loss, _ = staged_image_loss(outputs, clean - 0.5)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                prediction = torch.clamp(outputs["final"] + 0.5, 0.0, 1.0)
                psnr, mse = calc_psnr(prediction.detach(), clean)
                total_mse += mse
                total_psnr += psnr
                total_images += 1
            train_mse = total_mse / total_images
            train_psnr = total_psnr / total_images
            validation_mse, validation_psnr, noisy_psnr = evaluate(model, stage1_model, validation_loader, device)
            print(
                f"Epoch {epoch + 1}/{args.epochs} - Train MSE: {train_mse:.6f}, Train PSNR: {train_psnr:.2f} dB - "
                f"BSD68 MSE: {validation_mse:.6f}, PSNR: {validation_psnr:.2f} dB, Gain over noisy: {validation_psnr - noisy_psnr:+.2f} dB"
            )
            writer.add_scalar("train/mse_epoch", train_mse, epoch)
            writer.add_scalar("train/psnr_epoch", train_psnr, epoch)
            writer.add_scalar("bsd68/mse", validation_mse, epoch)
            writer.add_scalar("bsd68/psnr", validation_psnr, epoch)
            if train_mse < best_train_mse:
                best_train_mse = train_mse
                save_model(model, train_best_path)
            if validation_mse < best_validation_mse:
                best_validation_mse = validation_mse
                save_model(model, validation_best_path)
            if validation_psnr > best_validation_psnr:
                best_validation_psnr = validation_psnr
                save_model(model, validation_psnr_best_path)
            scheduler.step()
    finally:
        writer.close()


if __name__ == "__main__":
    main()
