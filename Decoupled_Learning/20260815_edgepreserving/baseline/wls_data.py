"""Single-channel paired WLS data used by the 20260818 edge-preserving task."""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


INPUT_SUFFIX = "-log"
TARGET_SUFFIX = "-WLS-l0p500-a1p400"


def _files_by_id(directory, suffix):
    files = {}
    for path in Path(directory).iterdir():
        if path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        if not path.stem.endswith(suffix):
            raise ValueError(f"Unexpected TIFF name in {directory}: {path.name}")
        identifier = path.stem[: -len(suffix)]
        if identifier in files:
            raise ValueError(f"Duplicate source ID {identifier} in {directory}")
        files[identifier] = path
    return files


class WLSPairs(Dataset):
    """Map source IDs to the original one-channel input and WLS target."""

    def __init__(self, data_root, split):
        if split not in {"train", "test"}:
            raise ValueError(f"Unknown WLS split: {split}")
        root = Path(data_root)
        tail = "" if split == "train" else "-200"
        inputs = _files_by_id(root / f"{split}input-l0p500-a1p400{tail}", INPUT_SUFFIX)
        targets = _files_by_id(root / f"{split}label-l0p500-a1p400{tail}", TARGET_SUFFIX)
        if not inputs or inputs.keys() != targets.keys():
            raise ValueError(
                f"Unmatched or empty {split} WLS pairs: "
                f"inputs={len(inputs)}, targets={len(targets)}, "
                f"missing_targets={sorted(inputs.keys() - targets.keys())[:5]}, "
                f"missing_inputs={sorted(targets.keys() - inputs.keys())[:5]}"
            )
        self.ids = sorted(inputs)
        self.pairs = [(inputs[identifier], targets[identifier]) for identifier in self.ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        input_path, target_path = self.pairs[index]
        image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        target = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
        if image is None or target is None or image.ndim != 2 or target.ndim != 2:
            raise ValueError(f"Expected one-channel WLS pair: {input_path}, {target_path}")
        if image.shape != target.shape:
            raise ValueError(f"WLS pair shape mismatch: {input_path}, {target_path}")
        image = image.astype(np.float32)
        target = target.astype(np.float32)
        if not np.isfinite(image).all() or not np.isfinite(target).all():
            raise ValueError(f"Nonfinite WLS pixels: {input_path}, {target_path}")
        # The source TIFFs are already normalized floats; do not recode to Y or RGB.
        return torch.from_numpy(np.ascontiguousarray(image[None])), torch.from_numpy(np.ascontiguousarray(target[None]))


def partition_ids(identifiers, val_count, seed):
    """Select a stable validation subset from train IDs, leaving test untouched."""
    if not 0 < val_count < len(identifiers):
        raise ValueError("val_count must be between one and train_count - 1")
    shuffled = sorted(identifiers)
    random.Random(seed).shuffle(shuffled)
    val_ids = set(shuffled[:val_count])
    return sorted(set(identifiers) - val_ids), sorted(val_ids)


def collate_pairs(batch):
    """Batch-minimum top-left crop followed by even-border trimming."""
    height = min(image.shape[-2] for image, _ in batch)
    width = min(image.shape[-1] for image, _ in batch)
    height -= height % 2
    width -= width % 2
    if height < 2 or width < 2:
        raise ValueError("WLS image is too small after even trimming")
    return (
        torch.stack([image[:, :height, :width] for image, _ in batch]),
        torch.stack([target[:, :height, :width] for _, target in batch]),
    )


def paired_patch_collate(batch, crop_size):
    """Use the RTV NAFNet run's random paired-patch geometry unchanged."""
    if crop_size == 0:
        return collate_pairs(batch)
    if crop_size < 0:
        raise ValueError("train_crop_size must be zero or positive")
    images, targets = [], []
    for image, target in batch:
        if image.shape != target.shape:
            raise ValueError(f"WLS pair shape mismatch: {image.shape} vs {target.shape}")
        height, width = image.shape[-2:]
        if height < crop_size:
            rows = torch.arange(crop_size, device=image.device).remainder(2 * height)
            rows = torch.where(rows < height, rows, 2 * height - 1 - rows)
            image = image.index_select(-2, rows)
            target = target.index_select(-2, rows)
            height = crop_size
        if width < crop_size:
            cols = torch.arange(crop_size, device=image.device).remainder(2 * width)
            cols = torch.where(cols < width, cols, 2 * width - 1 - cols)
            image = image.index_select(-1, cols)
            target = target.index_select(-1, cols)
            width = crop_size
        top = int(torch.randint(height - crop_size + 1, ()).item())
        left = int(torch.randint(width - crop_size + 1, ()).item())
        images.append(image[:, top : top + crop_size, left : left + crop_size])
        targets.append(target[:, top : top + crop_size, left : left + crop_size])
    return torch.stack(images), torch.stack(targets)
