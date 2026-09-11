import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def augment(image, mode):
    operations = (
        lambda x: x,
        np.flipud,
        np.rot90,
        lambda x: np.flipud(np.rot90(x)),
        lambda x: np.rot90(x, 2),
        lambda x: np.flipud(np.rot90(x, 2)),
        lambda x: np.rot90(x, 3),
        lambda x: np.flipud(np.rot90(x, 3)),
    )
    if mode not in range(8):
        raise ValueError(f"invalid augmentation mode: {mode}")
    return np.ascontiguousarray(operations[mode](image))


class OnlineGrayPatchDataset(Dataset):
    """Online noisy grayscale LR/HR patches and clean HR targets."""

    def __init__(self, image_dir, sigma=25.0, patch_size=40, stride=20, scale=2,
                 scales=(1.0, 0.9, 0.8, 0.7), train=False, seed=42,
                 augmentation=True):
        if sigma < 0 or patch_size <= 0 or stride <= 0 or scale <= 0:
            raise ValueError("invalid dataset parameters")
        if patch_size % scale or stride % scale:
            raise ValueError("patch_size and stride must be divisible by scale")
        self.sigma = float(sigma)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.scale = int(scale)
        self.scales = tuple(float(value) for value in scales)
        self.train = bool(train)
        self.seed = int(seed)
        self.augmentation = bool(augmentation and train)
        self.files = sorted(
            os.path.join(image_dir, name)
            for name in os.listdir(image_dir)
            if os.path.splitext(name)[1].lower() in VALID_EXTENSIONS
        )
        if not self.files:
            raise RuntimeError(f"No images found in {image_dir}")

        self.samples = []
        for file_index, path in enumerate(self.files):
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Failed to read image: {path}")
            height, width = image.shape
            for image_scale in self.scales:
                scaled_h, scaled_w = int(height * image_scale), int(width * image_scale)
                for top in range(0, scaled_h - patch_size + 1, stride):
                    for left in range(0, scaled_w - patch_size + 1, stride):
                        modes = range(8) if self.augmentation else (0,)
                        for mode in modes:
                            self.samples.append((file_index, image_scale, top, left, mode))
        if not self.samples:
            raise RuntimeError("no valid patches found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_index, image_scale, top, left, mode = self.samples[index]
        clean = cv2.imread(self.files[file_index], cv2.IMREAD_GRAYSCALE)
        if clean is None:
            raise RuntimeError(f"Failed to read image: {self.files[file_index]}")
        clean = clean.astype(np.float32) / 255.0
        rng = np.random.default_rng() if self.train else np.random.default_rng(self.seed + index * 1000003)
        noisy = clean + rng.normal(0.0, self.sigma / 255.0, clean.shape).astype(np.float32)

        height, width = clean.shape
        scaled_h, scaled_w = int(height * image_scale), int(width * image_scale)
        clean_scaled = cv2.resize(clean, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
        noisy_scaled = cv2.resize(noisy, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
        noisy_lr_image = cv2.resize(
            noisy_scaled,
            (max(1, scaled_w // self.scale), max(1, scaled_h // self.scale)),
            interpolation=cv2.INTER_LANCZOS4,
        )
        clean_hr = clean_scaled[top:top + self.patch_size, left:left + self.patch_size]
        noisy_hr = noisy_scaled[top:top + self.patch_size, left:left + self.patch_size]
        lr_size = self.patch_size // self.scale
        lr_top, lr_left = top // self.scale, left // self.scale
        noisy_lr = noisy_lr_image[lr_top:lr_top + lr_size, lr_left:lr_left + lr_size]
        return tuple(
            torch.from_numpy(augment(image, mode)[None]).float()
            for image in (noisy_lr, noisy_hr, clean_hr)
        )
