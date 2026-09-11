import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def augment_img(image, mode):
    operations = (
        lambda x: x,
        np.flipud,
        lambda x: np.rot90(x, 1),
        lambda x: np.flipud(np.rot90(x, 1)),
        lambda x: np.rot90(x, 2),
        lambda x: np.flipud(np.rot90(x, 2)),
        lambda x: np.rot90(x, 3),
        lambda x: np.flipud(np.rot90(x, 3)),
    )
    if mode not in range(8):
        raise ValueError(f"invalid augmentation mode: {mode}")
    return np.ascontiguousarray(operations[mode](image))


class DnCNNGrayLRPatchDataset(Dataset):
    """DnCNN-compatible grayscale pairs with an aligned LR input."""

    def __init__(self, image_dir, sigma=25.0, patch_size=192, scale=2,
                 train=False, seed=42, interpolation=cv2.INTER_LANCZOS4):
        if sigma < 0 or patch_size <= 0 or scale <= 0:
            raise ValueError("invalid dataset parameters")
        if patch_size % scale:
            raise ValueError("patch_size must be divisible by scale")

        self.image_dir = os.fspath(image_dir)
        self.sigma = float(sigma)
        self.patch_size = int(patch_size)
        self.scale = int(scale)
        self.train = bool(train)
        self.seed = int(seed)
        self.interpolation = int(interpolation)
        self.files = sorted(
            os.path.join(root, name)
            for root, _, names in os.walk(self.image_dir)
            for name in sorted(names)
            if os.path.splitext(name)[1].lower() in VALID_EXTENSIONS
        )
        if not self.files:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _read_gray(path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return image

    def _make_lr(self, noisy_hr):
        height, width = noisy_hr.shape[-2:]
        lr = cv2.resize(
            noisy_hr[0].numpy(),
            (width // self.scale, height // self.scale),
            interpolation=self.interpolation,
        )
        return torch.from_numpy(np.ascontiguousarray(lr[None])).float()

    def __getitem__(self, index):
        image = self._read_gray(self.files[index])

        if self.train:
            height, width = image.shape
            rnd_h = random.randint(0, max(0, height - self.patch_size))
            rnd_w = random.randint(0, max(0, width - self.patch_size))
            patch = image[
                rnd_h:rnd_h + self.patch_size,
                rnd_w:rnd_w + self.patch_size,
            ]
            patch = augment_img(patch, mode=random.randint(0, 7))
            clean_hr = torch.from_numpy(patch[None]).float().div_(255.0)
            noisy_hr = clean_hr.clone()
            noisy_hr.add_(torch.randn(noisy_hr.size()).mul_(self.sigma / 255.0))
        else:
            clean = image.astype(np.float32) / 255.0
            rng = np.random.RandomState(0)
            noisy = clean + rng.normal(
                0.0, self.sigma / 255.0, clean.shape
            ).astype(np.float32)
            clean_hr = torch.from_numpy(np.ascontiguousarray(clean[None])).float()
            noisy_hr = torch.from_numpy(np.ascontiguousarray(noisy[None])).float()

        noisy_lr = self._make_lr(noisy_hr)
        return noisy_lr, noisy_hr, clean_hr
