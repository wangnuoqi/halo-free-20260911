import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AdaptiveNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(1.0))
        self.w1 = nn.Parameter(torch.tensor(0.0))
        self.bn = nn.BatchNorm2d(channels, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w0 * x + self.w1 * self.bn(x)


def weights_init_identity(module):
    if isinstance(module, nn.Conv2d):
        nn.init.zeros_(module.weight)
        out_c, in_c, kh, kw = module.weight.shape
        for i in range(min(out_c, in_c)):
            module.weight.data[i, i, kh // 2, kw // 2] = 1.0
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class ClipartJpegDataset(Dataset):
    def __init__(self, data_root, quality, split, list_dir, patch_size=None):
        if patch_size is not None and (isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 2 or patch_size % 2):
            raise ValueError("patch_size must be an even integer >= 2 or None")
        self.patch_size = patch_size if split == "train" else None
        data_root = Path(data_root)
        self.jpeg_dir = data_root / f"jpeg_q{quality:02d}"
        self.clean_dir = data_root / "clean_png"
        list_path = Path(list_dir) / f"{split}_list.txt"
        stems = [x.strip() for x in list_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.pairs = []
        for stem in stems:
            jpg = self.jpeg_dir / f"{stem}.jpg"
            png = self.clean_dir / f"{stem}.png"
            if jpg.is_file() and png.is_file():
                self.pairs.append((jpg, png))
        if len(self.pairs) != len(stems):
            raise FileNotFoundError(f"Missing pairs in {list_path}: {len(stems)-len(self.pairs)}")
        print(f"{split}: {len(self.pairs)} pairs, quality={quality}")

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _read(path, channels):
        with Image.open(path) as image:
            image = image.convert("L" if channels == 1 else "RGB")
            arr = np.asarray(image, dtype=np.float32) / 255.0
        if channels == 1:
            arr = arr[None]
        else:
            arr = arr.transpose(2, 0, 1)
        return arr

    def __getitem__(self, index):
        jpg, png = self.pairs[index]
        x_hr = self._read(jpg, self.channels)
        gt = self._read(png, self.channels)
        if x_hr.shape != gt.shape:
            raise ValueError(f"Unaligned pair: {jpg} and {png}")
        if self.patch_size is not None:
            size = self.patch_size
            h, w = x_hr.shape[-2:]
            if min(h, w) < size:
                raise ValueError(f"Image {jpg} ({h}, {w}) smaller than patch {size}")
            top = random.randint(0, h - size)
            left = random.randint(0, w - size)
            x_hr = x_hr[:, top:top + size, left:left + size]
            gt = gt[:, top:top + size, left:left + size]
        if self.channels == 1:
            image_array = np.clip(x_hr[0] * 255, 0, 255).astype(np.uint8)
            image = Image.fromarray(image_array, mode="L")
        else:
            image_array = np.clip(x_hr.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            image = Image.fromarray(image_array, mode="RGB")
        lr_size = (max(1, image.width // 2), max(1, image.height // 2))
        x_lr = np.asarray(image.resize(lr_size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        if self.channels == 1:
            x_lr = x_lr[None]
        else:
            x_lr = x_lr.transpose(2, 0, 1)
        return torch.from_numpy(x_lr), torch.from_numpy(x_hr), torch.from_numpy(gt)

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, value):
        self._channels = value


def make_dataset(data_root, quality, split, list_dir, channels, patch_size=None):
    dataset = ClipartJpegDataset.__new__(ClipartJpegDataset)
    dataset._channels = channels
    ClipartJpegDataset.__init__(dataset, data_root, quality, split, list_dir, patch_size)
    return dataset


class ConvGuidedFilter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.box = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.conv_a = nn.Sequential(
            nn.Conv2d(2 * channels, 32, 1, bias=False), AdaptiveNorm(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1, bias=False), AdaptiveNorm(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 1, bias=False),
        )
        self.box.weight.data.fill_(1.0)

    def forward(self, x_lr, y_lr, x_hr):
        n = self.box(x_lr.new_ones(x_lr.shape))
        mean_x = self.box(x_lr) / n
        mean_y = self.box(y_lr) / n
        cov = self.box(x_lr * y_lr) / n - mean_x * mean_y
        var = self.box(x_lr * x_lr) / n - mean_x * mean_x
        a = self.conv_a(torch.cat([cov, var], dim=1))
        b = mean_y - a * mean_x
        a = F.interpolate(a, x_hr.shape[-2:], mode="bilinear", align_corners=True)
        b = F.interpolate(b, x_hr.shape[-2:], mode="bilinear", align_corners=True)
        return (a * x_hr + b).clamp(0, 1)


class DeepGuidedFilter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lr = nn.Sequential(
            nn.Conv2d(channels, 24, 3, padding=1, bias=False), AdaptiveNorm(24), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(24, 24, 3, padding=2, dilation=2, bias=False), AdaptiveNorm(24), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(24, 24, 3, padding=4, dilation=4, bias=False), AdaptiveNorm(24), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(24, 24, 3, padding=8, dilation=8, bias=False), AdaptiveNorm(24), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(24, 24, 3, padding=16, dilation=16, bias=False), AdaptiveNorm(24), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(24, 24, 3, padding=1, bias=False), AdaptiveNorm(24), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(24, channels, 1),
        )
        self.gf = ConvGuidedFilter(channels)
        self.apply(weights_init_identity)
        self.gf.box.weight.data.fill_(1.0)

    def forward(self, x_lr, x_hr):
        return self.gf(x_lr, self.lr(x_lr), x_hr)


def sobel_gradient_loss(pred, target):
    channels = pred.shape[1]
    kx = pred.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    ky = pred.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    pgx = F.conv2d(pred, kx, padding=1, groups=channels)
    pgy = F.conv2d(pred, ky, padding=1, groups=channels)
    tgx = F.conv2d(target, kx, padding=1, groups=channels)
    tgy = F.conv2d(target, ky, padding=1, groups=channels)
    return (pgx - tgx).abs().mean() + (pgy - tgy).abs().mean()


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_loss, args):
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({"epoch": epoch + 1, "model_state_dict": state, "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "best_loss": best_loss,
                "args": vars(args)}, path)
