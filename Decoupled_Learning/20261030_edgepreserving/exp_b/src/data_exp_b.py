"""Paired grayscale TIFF data pipeline for Exp B."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


DATA_DIRS = {
    "train": (
        "traininput-l0p500-a1p400_downsample",
        "traininput-l0p500-a1p400",
        "trainlabel-l0p500-a1p400",
    ),
    "test": (
        "testinput-l0p500-a1p400-200_downsample",
        "testinput-l0p500-a1p400-200",
        "testlabel-l0p500-a1p400-200",
    ),
}


def _index_files(directory: Path, label_files: bool = False) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(p for p in directory.iterdir() if p.is_file()):
        key = path.stem
        if label_files and "-WLS-l0p500-a1p400" in key:
            key = key.replace("-WLS-l0p500-a1p400", "-log", 1)
        if key in result:
            raise ValueError(f"duplicate image stem {key!r} in {directory}")
        result[key] = path
    return result


class PairedTripletDataset(Dataset):
    """Load aligned LR input, HR guide and HR target TIFF files by file stem."""

    def __init__(self, data_root: str | Path, split: str = "train"):
        if split not in DATA_DIRS:
            raise ValueError(f"split must be one of {tuple(DATA_DIRS)}, got {split!r}")
        self.data_root = Path(data_root)
        lr_name, hr_name, gt_name = DATA_DIRS[split]
        lr_by_id = _index_files(self.data_root / lr_name)
        hr_by_id = _index_files(self.data_root / hr_name)
        gt_by_id = _index_files(self.data_root / gt_name, label_files=True)
        if not lr_by_id or set(lr_by_id) != set(hr_by_id) or set(hr_by_id) != set(gt_by_id):
            missing_lr = sorted((set(hr_by_id) | set(gt_by_id)) - set(lr_by_id))
            missing_hr = sorted((set(lr_by_id) | set(gt_by_id)) - set(hr_by_id))
            missing_gt = sorted((set(lr_by_id) | set(hr_by_id)) - set(gt_by_id))
            raise ValueError(
                "LR/HR/GT file stems must match; "
                f"missing_lr={missing_lr[:5]}, missing_hr={missing_hr[:5]}, "
                f"missing_gt={missing_gt[:5]}"
            )
        self.ids = sorted(lr_by_id)
        self.paths = [(lr_by_id[key], hr_by_id[key], gt_by_id[key]) for key in self.ids]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        tensors = []
        for path in self.paths[index]:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"could not read TIFF image: {path}")
            if image.ndim != 2:
                raise ValueError(f"expected single-channel TIFF, got shape {image.shape}: {path}")
            image = np.asarray(image, dtype=np.float32)
            if not np.isfinite(image).all():
                raise ValueError(f"non-finite image values: {path}")
            tensors.append(torch.from_numpy(np.ascontiguousarray(image[None])))
        lr, hr, gt = tensors
        if hr.shape != gt.shape:
            raise ValueError(f"HR/GT dimensions differ for {self.ids[index]}: {hr.shape} vs {gt.shape}")
        return lr, hr, gt


class PairedPatchCollator:
    """Pickleable paired crop collator; crops HR origins on the LR scale grid."""

    def __init__(self, crop_size: int = 128, random_crop: bool = True):
        self.crop_size = int(crop_size)
        self.random_crop = bool(random_crop)
        if self.crop_size < 0:
            raise ValueError("crop_size must be zero (full image) or positive")

    def __call__(self, batch):
        if not batch:
            raise ValueError("cannot collate an empty batch")
        lr_batch, hr_batch, gt_batch = [], [], []
        for lr, hr, gt in batch:
            if lr.ndim != 3 or hr.ndim != 3 or gt.ndim != 3:
                raise ValueError("each LR/HR/GT tensor must have shape (1,H,W)")
            if hr.shape != gt.shape:
                raise ValueError(f"HR/GT dimensions differ: {hr.shape} vs {gt.shape}")
            h, w = hr.shape[-2:]
            lh, lw = lr.shape[-2:]
            if lh <= 0 or lw <= 0 or h % lh or w % lw or h // lh != w // lw:
                raise ValueError(
                    f"HR/LR scale must be equal and integral; HR={tuple(hr.shape)}, LR={tuple(lr.shape)}"
                )
            scale = h // lh
            if self.crop_size:
                crop = self.crop_size
                if crop % scale or crop > h or crop > w:
                    raise ValueError(
                        f"crop_size={crop} must fit the HR image and be divisible by scale={scale}"
                    )
                max_top = (h - crop) // scale
                max_left = (w - crop) // scale
                if self.random_crop:
                    top = random.randint(0, max_top) * scale
                    left = random.randint(0, max_left) * scale
                else:
                    top = (max_top // 2) * scale
                    left = (max_left // 2) * scale
                lr_crop = crop // scale
                lr_batch.append(lr[..., top // scale : top // scale + lr_crop,
                                   left // scale : left // scale + lr_crop])
                hr_batch.append(hr[..., top : top + crop, left : left + crop])
                gt_batch.append(gt[..., top : top + crop, left : left + crop])
            else:
                lr_batch.append(lr)
                hr_batch.append(hr)
                gt_batch.append(gt)

        if not self.crop_size and len(batch) > 1:
            scales = [hr.shape[-2] // lr.shape[-2] for lr, hr, _ in batch]
            if len(set(scales)) != 1:
                raise ValueError("full-image batches require a common HR/LR scale")
            scale = scales[0]
            height = min(hr.shape[-2] for hr in hr_batch)
            width = min(hr.shape[-1] for hr in hr_batch)
            height -= height % scale
            width -= width % scale
            lr_h, lr_w = height // scale, width // scale
            lr_batch = [lr[..., :lr_h, :lr_w] for lr in lr_batch]
            hr_batch = [hr[..., :height, :width] for hr in hr_batch]
            gt_batch = [gt[..., :height, :width] for gt in gt_batch]
        try:
            return torch.stack(lr_batch), torch.stack(hr_batch), torch.stack(gt_batch)
        except RuntimeError as exc:
            raise ValueError("images in a batch must have matching cropped dimensions") from exc


def deterministic_train_val_indices(
    identifiers: Sequence[str], val_count: int = 100, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Split sorted image IDs reproducibly; the independent test split is untouched."""
    count = len(identifiers)
    if not 0 < val_count < count:
        raise ValueError(f"val_count must be in [1, {count - 1}], got {val_count}")
    ordered = sorted(range(count), key=lambda index: identifiers[index])
    random.Random(seed).shuffle(ordered)
    val_indices = sorted(ordered[:val_count])
    val_set = set(val_indices)
    train_indices = [index for index in range(count) if index not in val_set]
    return train_indices, val_indices


class StridedEvalSampler(Sampler[int]):
    """Disjoint validation indices for each rank, without padding duplicates."""

    def __init__(self, data_source, rank: int, world_size: int):
        self.length = len(data_source)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self):
        return max(0, (self.length - self.rank + self.world_size - 1) // self.world_size)


def seed_worker(worker_id: int) -> None:
    """Seed Python/NumPy in workers from PyTorch's per-worker seed."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
