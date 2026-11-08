"""Dataset helpers for training refinement from precomputed diffusion outputs."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def _resolve_manifest_path(raw_path: str, cache_root: Path) -> Path:
    """Resolve Linux `/mnt/<drive>/...` paths in manifests on Windows too."""
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    normalized = raw_path.replace("\\", "/")
    parts = normalized.split("/")
    if len(parts) >= 4 and parts[1].lower() == "mnt" and len(parts[2]) == 1:
        windows_path = Path(f"{parts[2].upper()}:/" + "/".join(parts[3:]))
        if windows_path.exists():
            return windows_path

    # Keep manifests usable after the cache directory is moved as a unit.
    marker = "/dataset/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        relocated = cache_root.parent.parent / "dataset" / relative
        if relocated.exists():
            return relocated

    return candidate


def load_cache_entries(cache_root: str, manifest_name: str = "manifest.jsonl") -> List[dict]:
    """Load and validate manifest records plus their cached coarse outputs."""
    root = Path(cache_root)
    manifest_path = Path(manifest_name)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cache manifest does not exist: {manifest_path}")

    entries: List[dict] = []
    missing = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            input_path = _resolve_manifest_path(item["input"], root)
            target_path = _resolve_manifest_path(item["target"], root)
            prior_path = root / "images" / "syntheticprior" / f"{input_path.stem}.npy"
            coarse_path = root / "coarse" / f"{input_path.stem}_diff.png"
            required = {
                "input": input_path,
                "target": target_path,
                "prior": prior_path,
                "coarse": coarse_path,
            }
            absent = [name for name, path in required.items() if not path.is_file()]
            if absent:
                missing.append(
                    f"line {line_number}: missing {','.join(absent)} "
                    f"for {item.get('id', input_path.name)}"
                )
                continue
            entries.append(
                {
                    "id": item.get("id", input_path.stem),
                    "conditioning_image": str(input_path),
                    "image": str(target_path),
                    "prior": str(prior_path),
                    "coarse": str(coarse_path),
                }
            )

    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} cache entries are incomplete. First entries:\n{preview}"
        )
    if not entries:
        raise ValueError(f"Cache manifest contains no usable entries: {manifest_path}")
    return entries


def _load_prior_image(path: str) -> Image.Image:
    prior = np.load(path)
    if prior.ndim > 2:
        prior = prior.squeeze()
    prior = np.nan_to_num(prior, nan=0.0, posinf=1.0, neginf=0.0)
    prior = np.clip(prior.astype(np.float32), 0.0, 1.0)
    return Image.fromarray((prior * 255.0).astype(np.uint8))


class CachedRefineDataset(Dataset):
    """Load cond, cached coarse, GT, and prior with identical spatial transforms."""

    def __init__(
        self,
        entries: List[dict],
        resolution: int,
        resize_scale: float,
        disable_augment: bool,
    ):
        self.entries = entries
        self.resolution = resolution
        self.resize_scale = resize_scale
        self.disable_augment = disable_augment

    def __len__(self) -> int:
        return len(self.entries)

    def _resize(self, image: Image.Image, size: int) -> Image.Image:
        return image.resize((size, size), Image.Resampling.BILINEAR)

    def __getitem__(self, idx: int) -> dict:
        item = self.entries[idx]
        with (
            Image.open(item["conditioning_image"]) as cond_file,
            Image.open(item["coarse"]) as coarse_file,
            Image.open(item["image"]) as gt_file,
        ):
            cond = cond_file.convert("RGB")
            coarse = coarse_file.convert("RGB")
            gt = gt_file.convert("RGB")
        prior = _load_prior_image(item["prior"])

        if self.disable_augment:
            cond = self._resize(cond, self.resolution)
            coarse = self._resize(coarse, self.resolution)
            gt = self._resize(gt, self.resolution)
            prior = self._resize(prior, self.resolution)
        else:
            resize_size = int(self.resolution * self.resize_scale)
            if resize_size < self.resolution:
                raise ValueError("resize_scale must be >= 1.0 when augmentation is enabled.")
            cond = self._resize(cond, resize_size)
            coarse = self._resize(coarse, resize_size)
            gt = self._resize(gt, resize_size)
            prior = self._resize(prior, resize_size)

            i, j = torch.randint(0, resize_size - self.resolution + 1, (2,)).tolist()
            cond = TF.crop(cond, i, j, self.resolution, self.resolution)
            coarse = TF.crop(coarse, i, j, self.resolution, self.resolution)
            gt = TF.crop(gt, i, j, self.resolution, self.resolution)
            prior = TF.crop(prior, i, j, self.resolution, self.resolution)

            if random.random() < 0.5:
                cond = TF.hflip(cond)
                coarse = TF.hflip(coarse)
                gt = TF.hflip(gt)
                prior = TF.hflip(prior)
            if random.random() < 0.5:
                cond = TF.vflip(cond)
                coarse = TF.vflip(coarse)
                gt = TF.vflip(gt)
                prior = TF.vflip(prior)

        cond_tensor = TF.to_tensor(cond)
        coarse_tensor = TF.to_tensor(coarse)
        gt_tensor = TF.to_tensor(gt)
        prior_tensor = TF.to_tensor(prior)
        if not (
            cond_tensor.shape[-2:]
            == coarse_tensor.shape[-2:]
            == gt_tensor.shape[-2:]
            == prior_tensor.shape[-2:]
        ):
            raise ValueError(f"Cached sample has mismatched shapes: {item['id']}")
        return {
            "cond": cond_tensor,
            "prelim": coarse_tensor,
            "gt": gt_tensor,
            "prior": prior_tensor,
        }
