"""Fixed clean/JPEG pairs used by the local baseline training jobs."""
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class PairedJpegDataset(Dataset):
    def __init__(self, data_root, list_dir, quality, split, channels=3, patch_size=128, random_crop=True):
        self.root = Path(data_root)
        self.jpeg_dir = self.root / f"jpeg_q{int(quality):02d}"
        self.clean_dir = self.root / "clean_png"
        self.channels = channels
        self.patch_size = patch_size
        self.random_crop = random_crop
        stems = [x.strip() for x in (Path(list_dir) / f"{split}_list.txt").read_text().splitlines() if x.strip()]
        self.pairs = [(self.jpeg_dir / f"{s}.jpg", self.clean_dir / f"{s}.png") for s in stems]
        missing = [(a, b) for a, b in self.pairs if not a.is_file() or not b.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} pairs missing under {self.root}")

    def __len__(self):
        return len(self.pairs)

    def _read(self, path):
        mode = "L" if self.channels == 1 else "RGB"
        with Image.open(path) as image:
            return np.asarray(image.convert(mode), dtype=np.float32) / 255.0

    def __getitem__(self, index):
        low = self._read(self.pairs[index][0])
        high = self._read(self.pairs[index][1])
        if self.patch_size:
            h, w = low.shape[:2]
            size = min(self.patch_size, h, w)
            if self.random_crop:
                top = random.randint(0, h - size)
                left = random.randint(0, w - size)
            else:
                top, left = (h - size) // 2, (w - size) // 2
            low, high = low[top:top + size, left:left + size], high[top:top + size, left:left + size]
        if self.channels == 1:
            low, high = low[None], high[None]
        else:
            low, high = low.transpose(2, 0, 1), high.transpose(2, 0, 1)
        return torch.from_numpy(low.copy()), torch.from_numpy(high.copy())


def full_loader(data_root, list_dir, quality, split, channels=3):
    return PairedJpegDataset(data_root, list_dir, quality, split, channels, patch_size=None, random_crop=False)
