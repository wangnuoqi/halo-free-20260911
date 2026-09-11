"""Full-image training for the grayscale three-stage refinement network.

This script deliberately keeps the refine architecture, Sobel max normalization,
and cascade loss from the patch-training experiment unchanged.  It only changes
the data protocol: BSDS400 is trained and validated as whole images, with
geometry-only augmentation for training samples.
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from eval_convguided_gray import DeepGuidedFilter, load_model_weights
from eval_rf_3stagev3 import CascadeRefine3Stage, cascade_loss


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_DATA_ROOT = Path(r"D:\nuoqi\halo-free\data_bsds")
DEFAULT_EXPERIMENT_ROOT = Path(__file__).resolve().parent


def normalize_path_for_runtime(path: str | os.PathLike) -> str:
    path_string = str(path)
    if os.name == "nt":
        return path_string
    match = re.match(r"^([A-Za-z]):\\(.*)$", path_string)
    if not match:
        return path_string
    suffix = match.group(2).replace("\\", "/")
    return f"/mnt/{match.group(1).lower()}/{suffix}"


def list_image_paths(image_dir: str | os.PathLike) -> List[Path]:
    root = Path(normalize_path_for_runtime(image_dir))
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No supported images found in {root}")
    return paths


def crop_to_even(image: np.ndarray) -> np.ndarray:
    height = image.shape[0] - image.shape[0] % 2
    width = image.shape[1] - image.shape[1] % 2
    if height <= 0 or width <= 0:
        raise ValueError(f"Image is too small after even crop: {image.shape}")
    return np.ascontiguousarray(image[:height, :width])


def augment_geometry(image: np.ndarray) -> np.ndarray:
    rotation = random.randrange(4)
    augmented = np.rot90(image, k=rotation)
    if random.random() < 0.5:
        augmented = np.fliplr(augmented)
    if random.random() < 0.5:
        augmented = np.flipud(augmented)
    return np.ascontiguousarray(augmented)


class FullImageGrayDenoisingDataset(Dataset):
    """Whole-image grayscale AWGN data with optional geometry augmentation."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        sigma: float,
        scale: int = 2,
        augment: bool = False,
        deterministic_noise_seed: int | None = None,
        interpolation: int = cv2.INTER_LANCZOS4,
    ) -> None:
        if sigma < 0 or scale <= 0:
            raise ValueError("sigma must be non-negative and scale must be positive")
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        self.image_paths = list(image_paths)
        self.sigma = float(sigma)
        self.scale = int(scale)
        self.augment = bool(augment)
        self.deterministic_noise_seed = deterministic_noise_seed
        self.interpolation = interpolation

    def __len__(self) -> int:
        return len(self.image_paths)

    @staticmethod
    def _read_gray(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return image

    def _make_noise(self, shape: Tuple[int, int], index: int) -> np.ndarray:
        noise_scale = self.sigma / 255.0
        if self.deterministic_noise_seed is None:
            return np.random.normal(0.0, noise_scale, shape).astype(np.float32)
        rng = np.random.RandomState(self.deterministic_noise_seed + index)
        return rng.normal(0.0, noise_scale, shape).astype(np.float32)

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_path = self.image_paths[index]
        image = self._read_gray(image_path)
        if self.augment:
            image = augment_geometry(image)
        clean = crop_to_even(image).astype(np.float32) / 255.0
        noisy = clean + self._make_noise(clean.shape, index)

        height, width = clean.shape
        lr = cv2.resize(
            noisy,
            (width // self.scale, height // self.scale),
            interpolation=self.interpolation,
        )
        return {
            "noisy_lr": torch.from_numpy(np.ascontiguousarray(lr[None])).float(),
            "noisy_hr": torch.from_numpy(np.ascontiguousarray(noisy[None])).float(),
            "clean_hr": torch.from_numpy(np.ascontiguousarray(clean[None])).float(),
            "filename": image_path.name,
        }


def build_datasets(
    train_data_root: str | os.PathLike,
    validation_data_root: str | os.PathLike,
    sigma: float,
) -> Tuple[FullImageGrayDenoisingDataset, FullImageGrayDenoisingDataset]:
    """Build whole-image BSDS400 training and BSD68 validation datasets."""
    train_dataset = FullImageGrayDenoisingDataset(
        list_image_paths(train_data_root), sigma=sigma, augment=True
    )
    validation_dataset = FullImageGrayDenoisingDataset(
        list_image_paths(validation_data_root),
        sigma=sigma,
        augment=False,
        deterministic_noise_seed=0,
    )
    return train_dataset, validation_dataset


def calc_psnr(prediction: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    mse = torch.mean((prediction - target) ** 2).item()
    return (100.0 if mse == 0.0 else 10.0 * np.log10(1.0 / mse), mse)


def save_model(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(state_dict, path)
    print(f"Saved checkpoint: {path}")


def evaluate(
    model: torch.nn.Module,
    stage1_model: torch.nn.Module,
    loader: Iterable[Dict[str, object]],
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_mse = 0.0
    total_psnr = 0.0
    total_images = 0
    with torch.no_grad():
        for batch in loader:
            noisy_lr = batch["noisy_lr"].to(device, non_blocking=True)
            noisy_hr = batch["noisy_hr"].to(device, non_blocking=True)
            clean_hr = batch["clean_hr"].to(device, non_blocking=True)
            coarse = stage1_model(noisy_lr - 0.5, noisy_hr - 0.5)
            output = model(noisy_hr - 0.5, coarse - 0.5)["final"] + 0.5
            output = torch.clamp(output, 0.0, 1.0)
            psnr, mse = calc_psnr(output, clean_hr)
            batch_size = clean_hr.shape[0]
            total_mse += mse * batch_size
            total_psnr += psnr * batch_size
            total_images += batch_size
    return total_mse / total_images, total_psnr / total_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train grayscale CascadeRefine3Stage on full BSDS400 images."
    )
    parser.add_argument("--sigma", type=float, required=True, choices=(15.0, 25.0, 50.0))
    parser.add_argument("--train-data-root", default=str(DEFAULT_DATA_ROOT / "BSDS 400"))
    parser.add_argument("--validation-data-root", default=str(DEFAULT_DATA_ROOT / "BSD68"))
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--stage1-model-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Full-image BSDS samples have mixed shapes; use --batch-size 1")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")

    args.train_data_root = normalize_path_for_runtime(args.train_data_root)
    args.validation_data_root = normalize_path_for_runtime(args.validation_data_root)
    args.experiment_root = normalize_path_for_runtime(args.experiment_root)
    if args.stage1_model_path is not None:
        args.stage1_model_path = normalize_path_for_runtime(args.stage1_model_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    train_dataset, validation_dataset = build_datasets(
        args.train_data_root,
        args.validation_data_root,
        sigma=args.sigma,
    )
    print(
        f"Full-image datasets: train={len(train_dataset)} from {args.train_data_root}, "
        f"validation={len(validation_dataset)} from {args.validation_data_root}"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    stage1_path = args.stage1_model_path or str(
        Path(args.experiment_root)
        / "models"
        / f"convguided_bsds_sigma{args.sigma:g}"
        / f"convguided_bsds_sigma{args.sigma:g}_best.pth"
    )
    stage1_model = DeepGuidedFilter().to(device)
    load_model_weights(stage1_model, stage1_path, device)
    stage1_model.eval()
    for parameter in stage1_model.parameters():
        parameter.requires_grad_(False)

    model = CascadeRefine3Stage(
        base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    output_dir = (
        Path(args.experiment_root) / "models" / "refine_gray_fullimage" / f"sigma{args.sigma:g}"
    )
    train_best_path = output_dir / f"refine_gray_fullimage_sigma{args.sigma:g}_train_best.pth"
    validation_best_path = output_dir / f"refine_gray_fullimage_sigma{args.sigma:g}_val_best.pth"
    log_dir = (
        Path(args.experiment_root)
        / "log_dir"
        / f"refine_gray_fullimage_sigma{args.sigma:g}_{datetime.datetime.now():%Y%m%d-%H%M%S}"
    )
    writer = SummaryWriter(log_dir=str(log_dir))
    best_train_mse = float("inf")
    best_validation_mse = float("inf")

    try:
        for epoch in range(args.epochs):
            model.train()
            total_train_mse = 0.0
            total_train_psnr = 0.0
            total_images = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
                noisy_lr = batch["noisy_lr"].to(device, non_blocking=True)
                noisy_hr = batch["noisy_hr"].to(device, non_blocking=True)
                clean_hr = batch["clean_hr"].to(device, non_blocking=True)
                with torch.no_grad():
                    coarse = stage1_model(noisy_lr - 0.5, noisy_hr - 0.5)

                outputs = model(noisy_hr - 0.5, coarse - 0.5)
                loss, _ = cascade_loss(
                    outputs,
                    coarse - 0.5,
                    clean_hr - 0.5,
                    thresh_mode="otsu",
                    dilate=0,
                    loss_type="l1",
                    normalize_grad=True,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                output = torch.clamp(outputs["final"] + 0.5, 0.0, 1.0)
                psnr, mse = calc_psnr(output.detach(), clean_hr)
                batch_size = clean_hr.shape[0]
                total_train_mse += mse * batch_size
                total_train_psnr += psnr * batch_size
                total_images += batch_size

            train_mse = total_train_mse / total_images
            train_psnr = total_train_psnr / total_images
            validation_mse, validation_psnr = evaluate(
                model, stage1_model, validation_loader, device
            )
            print(
                f"Epoch {epoch + 1}/{args.epochs} - "
                f"Train MSE: {train_mse:.6f}, Train PSNR: {train_psnr:.2f} dB - "
                f"Val MSE: {validation_mse:.6f}, Val PSNR: {validation_psnr:.2f} dB"
            )
            writer.add_scalar("train/mse_epoch", train_mse, epoch)
            writer.add_scalar("train/psnr_epoch", train_psnr, epoch)
            writer.add_scalar("validation/mse", validation_mse, epoch)
            writer.add_scalar("validation/psnr", validation_psnr, epoch)

            if train_mse < best_train_mse:
                best_train_mse = train_mse
                save_model(model, train_best_path)
            if validation_mse < best_validation_mse:
                best_validation_mse = validation_mse
                save_model(model, validation_best_path)
            scheduler.step()
    finally:
        writer.close()


if __name__ == "__main__":
    main()
