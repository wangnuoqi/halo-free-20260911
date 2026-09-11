from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


VALID_EXTENSIONS = {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _sync_augment(rgb, depth_lr, depth_hr):
    if np.random.rand() < 0.5:
        rgb = rgb[:, :, ::-1].copy()
        depth_lr = depth_lr[:, :, ::-1].copy()
        depth_hr = depth_hr[:, :, ::-1].copy()
    if np.random.rand() < 0.5:
        rgb = rgb[:, ::-1, :].copy()
        depth_lr = depth_lr[:, ::-1, :].copy()
        depth_hr = depth_hr[:, ::-1, :].copy()
    # Keep non-square full images stackable when using batch_size > 1.
    # NYUv2 is 480x640, so a transpose would create mixed 480x640/640x480
    # shapes inside one batch. Square inputs can still use this augmentation.
    if rgb.shape[1] == rgb.shape[2] and np.random.rand() < 0.5:
        rgb = np.transpose(rgb, (0, 2, 1)).copy()
        depth_lr = np.transpose(depth_lr, (0, 2, 1)).copy()
        depth_hr = np.transpose(depth_hr, (0, 2, 1)).copy()
    return rgb, depth_lr, depth_hr


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    return np.transpose(rgb / 255.0, (2, 0, 1))


def _read_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        with Image.open(path) as image:
            depth = np.asarray(image)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise RuntimeError(f"Expected single-channel depth image: {path}")
    depth = depth.astype(np.float32, copy=False)
    if not np.isfinite(depth).all():
        raise RuntimeError(f"Non-finite depth values found in {path}")
    return depth


class NYUv2FullImageDataset(Dataset):
    """Full-image NYUv2 loader with per-image depth normalization."""

    def __init__(
        self,
        data_root,
        split="train",
        scale=4,
        indices: Sequence[int] | None = None,
        augment=False,
        depth_unit_scale=1.0,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.scale = int(scale)
        self.augment = bool(augment)
        self.depth_unit_scale = float(depth_unit_scale)
        if self.scale < 1:
            raise ValueError(f"scale must be positive, got {scale}")

        self.rgb_dir = self.data_root / "image" / split
        self.depth_dir = self.data_root / "depth_label" / split
        if not self.rgb_dir.is_dir() or not self.depth_dir.is_dir():
            fallback_dirs = {
                "train": (self.data_root / "RGB_train", self.data_root / "depth_train"),
                "test": (self.data_root / "nyu_rgb_test", self.data_root / "nyu_test_gt_16bit"),
            }
            self.rgb_dir, self.depth_dir = fallback_dirs.get(
                split, (self.rgb_dir, self.depth_dir)
            )
        if not self.rgb_dir.is_dir() or not self.depth_dir.is_dir():
            raise FileNotFoundError(
                f"Expected image/depth_label directories for split={split} under {self.data_root}"
            )

        rgb_names = {
            path.stem: path
            for path in self.rgb_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        }
        depth_names = {
            path.stem: path
            for path in self.depth_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        }
        names = sorted(rgb_names.keys() & depth_names.keys(), key=self._sort_key)
        if indices is not None:
            names = [names[index] for index in indices]
        if not names:
            raise RuntimeError(f"No paired RGB/depth images found under {self.data_root}")

        self.names = names
        self.rgb_files = [rgb_names[name] for name in names]
        self.depth_files = [depth_names[name] for name in names]

    @staticmethod
    def _sort_key(name):
        return (0, int(name)) if name.isdigit() else (1, name)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        rgb = _read_rgb(self.rgb_files[index])
        depth_raw = _read_depth(self.depth_files[index]) * self.depth_unit_scale
        if rgb.shape[1:] != depth_raw.shape:
            raise RuntimeError(
                f"RGB/depth shape mismatch for {self.names[index]}: "
                f"RGB={rgb.shape}, depth={depth_raw.shape}"
            )

        height_cropped = depth_raw.shape[0] - depth_raw.shape[0] % self.scale
        width_cropped = depth_raw.shape[1] - depth_raw.shape[1] % self.scale
        if height_cropped < 1 or width_cropped < 1:
            raise RuntimeError(f"Image is too small for x{self.scale}: {depth_raw.shape}")
        depth_raw = depth_raw[:height_cropped, :width_cropped]
        rgb = rgb[:, :height_cropped, :width_cropped]

        depth_min = float(depth_raw.min())
        depth_max = float(depth_raw.max())
        depth_range = depth_max - depth_min
        if not np.isfinite(depth_range) or depth_range <= 0.0:
            raise RuntimeError(f"Depth image has no finite range: {self.names[index]}")

        depth_hr = ((depth_raw - depth_min) / depth_range).astype(np.float32)
        depth_lr = np.array(
            Image.fromarray(depth_hr).resize(
                (width_cropped // self.scale, height_cropped // self.scale),
                Image.Resampling.BICUBIC,
            ),
            dtype=np.float32,
            copy=True,
        )
        depth_lr = depth_lr[None, :, :]
        depth_hr = depth_hr[None, :, :]

        if self.augment:
            rgb, depth_lr, depth_hr = _sync_augment(rgb, depth_lr, depth_hr)

        return (
            torch.from_numpy(np.ascontiguousarray(depth_lr)),
            torch.from_numpy(np.ascontiguousarray(rgb)),
            torch.from_numpy(np.ascontiguousarray(depth_hr)),
            torch.tensor(depth_min, dtype=torch.float32),
            torch.tensor(depth_max, dtype=torch.float32),
            self.names[index],
        )
