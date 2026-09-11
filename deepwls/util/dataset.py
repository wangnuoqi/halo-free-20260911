import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


VALID_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def _collect_images(directory: Path):
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )


def _sample_id_from_input(path: Path):
    name = path.name
    if "-input" not in name:
        raise ValueError(f"Unsupported input filename: {name}")
    sample_id = name.split("-input", 1)[0]
    sample_path = Path(sample_id)
    if sample_path.suffix.lower() in VALID_SUFFIXES:
        return sample_path.stem
    return sample_id


def _expected_label_names(input_path: Path, sample_id: str, split: str):
    if split == "train":
        return [
            f"{sample_id}-WLS-025000.tiff",
            f"{input_path.stem}-WLS-025000.tiff",
        ]
    if split == "test":
        return [f"{sample_id}.tiff-WLS-025000.tiff"]
    raise ValueError(f"Unsupported split: {split}")


def build_paired_samples(input_dir, label_dir, split):
    input_dir = Path(input_dir)
    label_dir = Path(label_dir)

    samples = []
    for input_path in _collect_images(input_dir):
        sample_id = _sample_id_from_input(input_path)
        label_path = None
        for candidate_name in _expected_label_names(input_path, sample_id, split):
            candidate_path = label_dir / candidate_name
            if candidate_path.exists():
                label_path = candidate_path
                break
        if label_path is None:
            raise FileNotFoundError(f"Missing label for sample {sample_id} under {label_dir}")
        samples.append((input_path, label_path, sample_id))

    if not samples:
        raise RuntimeError(f"No paired samples found under {input_dir}")
    return samples


def _load_grayscale_tensor(path: Path):
    array = np.array(Image.open(path), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected grayscale image at {path}, got shape {array.shape}")
    tensor = torch.from_numpy(array).unsqueeze(0)
    return tensor.repeat(3, 1, 1)


def _crop_pair(input_tensor, label_tensor, top, left, patch_size):
    bottom = top + patch_size
    right = left + patch_size
    return (
        input_tensor[:, top:bottom, left:right],
        label_tensor[:, top:bottom, left:right],
    )


class PairedImageDataset(Dataset):
    def __init__(self, input_dir, label_dir, split, patch_size=None):
        self.samples = build_paired_samples(input_dir, label_dir, split)
        self.split = split
        self.patch_size = patch_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        input_path, label_path, sample_id = self.samples[index]
        input_tensor = _load_grayscale_tensor(input_path)
        label_tensor = _load_grayscale_tensor(label_path)

        if input_tensor.shape != label_tensor.shape:
            raise ValueError(
                f"Mismatched shapes for {sample_id}: input={tuple(input_tensor.shape)} label={tuple(label_tensor.shape)}"
            )

        if self.patch_size is not None:
            _, height, width = input_tensor.shape
            if height < self.patch_size or width < self.patch_size:
                raise ValueError(
                    f"Patch size {self.patch_size} exceeds sample size {(height, width)} for {sample_id}"
                )
            if self.split == "train":
                top = random.randint(0, height - self.patch_size)
                left = random.randint(0, width - self.patch_size)
            else:
                top = max((height - self.patch_size) // 2, 0)
                left = max((width - self.patch_size) // 2, 0)
            input_tensor, label_tensor = _crop_pair(input_tensor, label_tensor, top, left, self.patch_size)

        return input_tensor.contiguous(), label_tensor.contiguous(), sample_id


class SingleImageDataset(Dataset):
    def __init__(self, input_dir, patch_size=None):
        self.input_dir = Path(input_dir)
        self.patch_size = patch_size
        self.samples = []
        for input_path in _collect_images(self.input_dir):
            sample_id = _sample_id_from_input(input_path)
            self.samples.append((input_path, sample_id))

        if not self.samples:
            raise RuntimeError(f"No input samples found under {self.input_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        input_path, sample_id = self.samples[index]
        input_tensor = _load_grayscale_tensor(input_path)

        if self.patch_size is not None:
            _, height, width = input_tensor.shape
            if height < self.patch_size or width < self.patch_size:
                raise ValueError(
                    f"Patch size {self.patch_size} exceeds sample size {(height, width)} for {sample_id}"
                )
            top = random.randint(0, height - self.patch_size)
            left = random.randint(0, width - self.patch_size)
            bottom = top + self.patch_size
            right = left + self.patch_size
            input_tensor = input_tensor[:, top:bottom, left:right]

        return input_tensor.contiguous(), sample_id
