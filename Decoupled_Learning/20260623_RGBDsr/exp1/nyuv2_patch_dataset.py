from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def synchronized_augment(rgb_hr, depth_lr, depth_hr):
    """Apply identical spatial operations to the three aligned CHW arrays."""
    if random.random() < 0.5:
        rgb_hr = rgb_hr[:, :, ::-1].copy()
        depth_lr = depth_lr[:, :, ::-1].copy()
        depth_hr = depth_hr[:, :, ::-1].copy()

    if random.random() < 0.5:
        rgb_hr = rgb_hr[:, ::-1, :].copy()
        depth_lr = depth_lr[:, ::-1, :].copy()
        depth_hr = depth_hr[:, ::-1, :].copy()

    if random.random() < 0.5:
        rgb_hr = np.transpose(rgb_hr, (0, 2, 1)).copy()
        depth_lr = np.transpose(depth_lr, (0, 2, 1)).copy()
        depth_hr = np.transpose(depth_hr, (0, 2, 1)).copy()

    return rgb_hr, depth_lr, depth_hr


class NYUv2DepthPatchDataset(Dataset):
    def __init__(self, patch_root, depth_max=None, augment=False, return_depth_stats=False):
        self.patch_root = Path(patch_root)
        self.depth_lr_dir = self.patch_root / "DepthLr"
        self.rgb_dir = self.patch_root / "RGB"
        self.depth_hr_dir = self.patch_root / "DepthHR"
        self.augment = bool(augment)
        self.return_depth_stats = bool(return_depth_stats)

        for folder in (self.depth_lr_dir, self.rgb_dir, self.depth_hr_dir):
            if not folder.is_dir():
                raise RuntimeError(f"Missing patch folder: {folder}")

        self.names = self._find_common_names()
        if not self.names:
            raise RuntimeError(f"No matching DepthLr/RGB/DepthHR patch triplets found in {self.patch_root}")

        print(
            f"NYUv2DepthPatchDataset: {len(self.names)} patches from {self.patch_root} "
            f"(augment={self.augment})"
        )

    def _find_common_names(self):
        lr_names = {path.stem for path in self.depth_lr_dir.glob("*.npy")}
        rgb_names = {path.stem for path in self.rgb_dir.glob("*.npy")}
        hr_names = {path.stem for path in self.depth_hr_dir.glob("*.npy")}
        common = lr_names & rgb_names & hr_names
        return sorted(common, key=self._sort_key)

    @staticmethod
    def _sort_key(name):
        return int(name) if name.isdigit() else name

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        depth_lr = np.load(self.depth_lr_dir / f"{name}.npy").astype(np.float32)
        rgb_hr = np.load(self.rgb_dir / f"{name}.npy").astype(np.float32)
        depth_hr = np.load(self.depth_hr_dir / f"{name}.npy").astype(np.float32)

        if depth_lr.ndim != 3 or depth_hr.ndim != 3 or rgb_hr.ndim != 3:
            raise RuntimeError(
                f"Expected CHW npy patches for {name}: "
                f"DepthLr={depth_lr.shape}, RGB={rgb_hr.shape}, DepthHR={depth_hr.shape}"
            )
        if depth_lr.shape[0] != 1 or depth_hr.shape[0] != 1 or rgb_hr.shape[0] != 3:
            raise RuntimeError(
                f"Unexpected channel count for {name}: "
                f"DepthLr={depth_lr.shape}, RGB={rgb_hr.shape}, DepthHR={depth_hr.shape}"
            )
        if rgb_hr.shape[1:] != depth_hr.shape[1:]:
            raise RuntimeError(
                f"RGB and DepthHR spatial sizes must match for {name}: "
                f"RGB={rgb_hr.shape}, DepthHR={depth_hr.shape}"
            )
        if not np.isfinite(depth_lr).all() or not np.isfinite(depth_hr).all():
            raise RuntimeError(f"Non-finite depth values found in patch {name}")

        rgb_hr = rgb_hr / 255.0
        min_d = float(depth_hr.min())
        max_d = float(depth_hr.max())
        depth_range = max(max_d - min_d, 1e-6)

        # Both depth resolutions share the HR patch's physical range.
        depth_hr = np.clip((depth_hr - min_d) / depth_range, 0.0, 1.0)
        depth_lr = np.clip((depth_lr - min_d) / depth_range, 0.0, 1.0)

        if self.augment:
            rgb_hr, depth_lr, depth_hr = synchronized_augment(
                rgb_hr, depth_lr, depth_hr
            )

        sample = (
            torch.from_numpy(depth_lr),
            torch.from_numpy(rgb_hr),
            torch.from_numpy(depth_hr),
        )
        if self.return_depth_stats:
            return sample + (
                torch.tensor(min_d, dtype=torch.float32),
                torch.tensor(max_d, dtype=torch.float32),
            )
        return sample
