from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".tif", ".tiff"}
RTV_LAMBDA = 0.01


def encode_rtv_condition(lambda_value: float = RTV_LAMBDA) -> float:
    """Apply the RTV lambda mapping used by the official single-operator script."""
    return (float(lambda_value) - 0.002) * 4.125 + 0.002


def _read_rgb_255(path: Path) -> torch.Tensor:
    """Read RGB TIFF data in the BGR tensor order used by the three-stage loader."""
    image = np.asarray(tifffile.imread(path))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB HWC TIFF at {path}, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / np.iinfo(image.dtype).max
    else:
        image = image.astype(np.float32)
        if not np.isfinite(image).all():
            raise ValueError(f"Non-finite pixel values in {path}")
        if image.max(initial=0.0) <= 1.0:
            pass
        elif image.max(initial=0.0) <= 255.0:
            image = image / 255.0
        else:
            raise ValueError(f"Expected normalized or 8-bit-range RGB data at {path}")
    image = np.clip(image, 0.0, 1.0)[..., ::-1] * 255.0
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


class RTVPairDataset(Dataset):
    """Read paired RGB TIFFs under one split's input/ and label/ directories."""

    def __init__(self, split_root):
        self.split_root = Path(split_root)
        input_dir = self.split_root / "input"
        label_dir = self.split_root / "label"
        inputs = {p.stem: p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
        labels = {p.stem: p for p in label_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
        if not inputs or not labels:
            raise RuntimeError(f"No RGB TIFF pairs found under {self.split_root}")
        if inputs.keys() != labels.keys():
            missing_labels = sorted(inputs.keys() - labels.keys())
            missing_inputs = sorted(labels.keys() - inputs.keys())
            raise RuntimeError(
                f"Unmatched TIFF pairs under {self.split_root}: "
                f"missing labels={missing_labels[:5]}, missing inputs={missing_inputs[:5]}"
            )
        self.pairs = [(inputs[stem], labels[stem]) for stem in sorted(inputs)]
        print(f"{self.split_root}: {len(self.pairs)} RGB input/label pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_path, label_path = self.pairs[index]
        image = _read_rgb_255(input_path)
        label = _read_rgb_255(label_path)
        if image.shape != label.shape:
            raise ValueError(f"Pair shape mismatch: {input_path} {image.shape}, {label_path} {label.shape}")
        return image, label


def build_model_input(image_255: torch.Tensor) -> torch.Tensor:
    """Create BGR, edge channels and apply the official 128 input centering."""
    if image_255.ndim != 4 or image_255.shape[1] != 3:
        raise ValueError(f"Expected [B,3,H,W] RGB input, got {tuple(image_255.shape)}")
    diff_x = F.pad((image_255[:, :, :, 1:] - image_255[:, :, :, :-1]).abs(), (1, 0, 0, 0))
    diff_x[:, :, :, :-1] += (image_255[:, :, :, 1:] - image_255[:, :, :, :-1]).abs()
    diff_y = F.pad((image_255[:, :, 1:, :] - image_255[:, :, :-1, :]).abs(), (0, 0, 1, 0))
    diff_y[:, :, :-1, :] += (image_255[:, :, 1:, :] - image_255[:, :, :-1, :]).abs()
    edge = (diff_x + diff_y).sum(dim=1, keepdim=True) / 12.0
    return torch.cat((image_255, edge), dim=1) - 128.0


def paired_crop_collate(batch: Sequence, crop_size: Optional[int], random_crop: bool):
    height = min(min(image.shape[-2], label.shape[-2]) for image, label in batch)
    width = min(min(image.shape[-1], label.shape[-1]) for image, label in batch)
    crop_h = height if crop_size is None else min(crop_size, height)
    crop_w = width if crop_size is None else min(crop_size, width)
    crop_h -= crop_h % 2
    crop_w -= crop_w % 2
    images, labels = [], []
    for image, label in batch:
        image_h, image_w = image.shape[-2:]
        if crop_size is None:
            top, left = 0, 0
        elif random_crop:
            top = int(torch.randint(image_h - crop_h + 1, ()).item())
            left = int(torch.randint(image_w - crop_w + 1, ()).item())
        else:
            top, left = (image_h - crop_h) // 2, (image_w - crop_w) // 2
        images.append(image[:, top : top + crop_h, left : left + crop_w])
        labels.append(label[:, top : top + crop_h, left : left + crop_w])
    return torch.stack(images), torch.stack(labels)
