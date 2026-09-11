#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dncnn_gray_lr_patch_dataset import DnCNNGrayLRPatchDataset


CHANNELS = 1
DEFAULT_EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = r"D:\nuoqi\halo-free\data_bsds\CBSD68"
DEFAULT_MODEL_PATH = (
    Path(DEFAULT_EXPERIMENT_ROOT)
    / "models"
    / "convguided_bsds_sigma25"
    / "convguided_bsds_sigma25_best.pth"
)


class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))
        self.bn = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1, channels=CHANNELS):
        super().__init__()
        self.channels = channels
        self.box_filter = nn.Conv2d(
            channels, channels, kernel_size=3, padding=radius,
            dilation=radius, bias=False, groups=channels,
        )
        self.conv_a = nn.Sequential(
            nn.Conv2d(channels * 2, 32, kernel_size=1, bias=False),
            AdaptiveNorm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            AdaptiveNorm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=1, bias=False),
        )
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):
        _, channels, h_lr, w_lr = x_lr.size()
        _, _, h_hr, w_hr = x_hr.size()
        if channels != self.channels:
            raise RuntimeError(f"Expected {self.channels} input channels, got {channels}")

        n = self.box_filter(x_lr.new_ones((1, self.channels, h_lr, w_lr)))
        mean_x = self.box_filter(x_lr) / n
        mean_y = self.box_filter(y_lr) / n
        cov_xy = self.box_filter(x_lr * y_lr) / n - mean_x * mean_y
        var_x = self.box_filter(x_lr * x_lr) / n - mean_x * mean_x
        a = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        b = mean_y - a * mean_x
        mean_a = F.interpolate(a, (h_hr, w_hr), mode="bilinear", align_corners=True)
        mean_b = F.interpolate(b, (h_hr, w_hr), mode="bilinear", align_corners=True)
        return mean_a * x_hr + mean_b


def build_lr_net(layer=5):
    layers = [
        nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
        AdaptiveNorm(24),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    for level in range(1, layer):
        layers += [
            nn.Conv2d(24, 24, kernel_size=3, padding=2 ** level,
                      dilation=2 ** level, bias=False),
            AdaptiveNorm(24),
            nn.LeakyReLU(0.2, inplace=True),
        ]
    layers += [
        nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False),
        AdaptiveNorm(24),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(24, 1, kernel_size=1),
    ]
    return nn.Sequential(*layers)


class DeepGuidedFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.lr = build_lr_net()
        self.gf = ConvGuidedFilter()

    def forward(self, x_lr, x_hr):
        return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)


def load_model_weights(model, path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def crop_to_even(tensor):
    _, _, height, width = tensor.shape
    return tensor[:, :, :height - height % 2, :width - width % 2]


def metrics_path_for(output_dir, sigma):
    output_dir = Path(output_dir)
    return output_dir / f"metrics_sigma{float(sigma):g}.json"


def output_dir_for_sigma(output_root, sigma):
    """Return the per-noise-level directory used for saved evaluation outputs."""
    return Path(output_root) / f"sigma{float(sigma):g}"


def save_prediction(prediction, output_path):
    """Save one grayscale prediction tensor in the same 8-bit format as the inputs."""
    image = prediction.detach().cpu().squeeze(0).numpy()
    image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write prediction: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the grayscale ConvGuidedFilter model.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory; defaults to eval_results-img/sigma<sigma>.",
    )
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--downsample-scale", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    dataset = DnCNNGrayLRPatchDataset(
        args.data_root,
        sigma=args.sigma,
        scale=args.downsample_scale,
        train=False,
        seed=0,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = DeepGuidedFilter().to(device)
    load_model_weights(model, args.model_path, device)
    model.eval()

    total_mse = 0.0
    total_psnr = 0.0
    output_dir = Path(args.output_dir) if args.output_dir else output_dir_for_sigma(
        Path(DEFAULT_EXPERIMENT_ROOT) / "eval_results-img", args.sigma
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index, (x_lr, x_hr, label) in enumerate(tqdm(loader, desc="Evaluating", dynamic_ncols=True)):
            x_hr = crop_to_even(x_hr)
            label = crop_to_even(label)
            x_lr = x_lr.to(device) - 0.5
            x_hr = x_hr.to(device) - 0.5
            label = label.to(device)
            prediction = model(x_lr, x_hr)

            # Evaluation uses batch_size=1, so the dataset file name remains stable.
            save_prediction(prediction[0], output_dir / dataset.files[index].split(os.sep)[-1])

            mse = torch.mean((prediction - label) ** 2).item()
            total_mse += mse
            total_psnr += 10.0 * np.log10(1.0 / max(mse, 1e-12))
    mean_mse = total_mse / len(dataset)
    mean_psnr = total_psnr / len(dataset)
    metrics = {
        "data_root": str(args.data_root),
        "model_path": str(args.model_path),
        "sigma": args.sigma,
        "downsample_scale": args.downsample_scale,
        "samples": len(dataset),
        "mse": mean_mse,
        "psnr": mean_psnr,
    }
    metrics_path = metrics_path_for(output_dir, args.sigma)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Data root: {args.data_root}")
    print(f"Model path: {args.model_path}")
    print(f"Samples: {len(dataset)}")
    print(f"MSE: {mean_mse:.8f}")
    print(f"PSNR: {mean_psnr:.4f} dB")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
