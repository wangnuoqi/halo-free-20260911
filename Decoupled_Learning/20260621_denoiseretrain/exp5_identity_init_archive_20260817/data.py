from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_image_paths(image_dir: str | os.PathLike) -> List[Path]:
    root = Path(image_dir)
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
    augmented = np.rot90(image, k=random.randrange(4))
    if random.random() < 0.5:
        augmented = np.fliplr(augmented)
    if random.random() < 0.5:
        augmented = np.flipud(augmented)
    return np.ascontiguousarray(augmented)


class FullImageGrayDenoisingDataset(Dataset):
    """Whole-image gray AWGN samples with optional geometry-only augmentation."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        sigma: float,
        augment: bool,
        scale: int = 2,
        deterministic_noise_seed: int | None = None,
    ) -> None:
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        self.image_paths = list(image_paths)
        self.sigma = float(sigma)
        self.augment = bool(augment)
        self.scale = int(scale)
        self.deterministic_noise_seed = deterministic_noise_seed

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_path = self.image_paths[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        if self.augment:
            image = augment_geometry(image)
        clean = crop_to_even(image).astype(np.float32) / 255.0
        if self.deterministic_noise_seed is None:
            noise = np.random.normal(0.0, self.sigma / 255.0, clean.shape)
        else:
            rng = np.random.RandomState(self.deterministic_noise_seed)
            noise = rng.normal(0.0, self.sigma / 255.0, clean.shape)
        noisy = clean + noise.astype(np.float32)
        height, width = clean.shape
        noisy_lr = cv2.resize(
            noisy,
            (width // self.scale, height // self.scale),
            interpolation=cv2.INTER_LANCZOS4,
        )
        return {
            "noisy_lr": torch.from_numpy(np.ascontiguousarray(noisy_lr[None])).float(),
            "noisy_hr": torch.from_numpy(np.ascontiguousarray(noisy[None])).float(),
            "clean_hr": torch.from_numpy(np.ascontiguousarray(clean[None])).float(),
            "filename": image_path.name,
        }


def build_datasets(
    train_data_root: str | os.PathLike,
    validation_data_root: str | os.PathLike,
    sigma: float,
) -> Tuple[FullImageGrayDenoisingDataset, FullImageGrayDenoisingDataset]:
    return (
        FullImageGrayDenoisingDataset(list_image_paths(train_data_root), sigma, augment=True),
        FullImageGrayDenoisingDataset(
            list_image_paths(validation_data_root),
            sigma,
            augment=False,
            deterministic_noise_seed=0,
        ),
    )
