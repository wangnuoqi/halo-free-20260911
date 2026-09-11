#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CHANNELS = 3
VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_DATA_ROOT = Path("/mnt/d/nuoqi/halo-free/data_bsds/traindata_sigma25")
DEFAULT_EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = DEFAULT_EXPERIMENT_ROOT / "models" / "convguided_bsds_sigma25" / "convguided_bsds_sigma25_best.pth"


class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))
        self.bn = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


def weights_init_identity(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        n_out, n_in, h, w = m.weight.data.size()
        if n_out < n_in:
            nn.init.xavier_uniform_(m.weight.data)
            return
        m.weight.data.zero_()
        ch, cw = h // 2, w // 2
        for i in range(n_in):
            m.weight.data[i, i, ch, cw] = 1.0
    elif classname.find("BatchNorm2d") != -1:
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)


class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1, channels=CHANNELS, norm=nn.BatchNorm2d):
        super().__init__()
        self.channels = channels
        self.box_filter = nn.Conv2d(
            channels, channels, kernel_size=3, padding=radius, dilation=radius,
            bias=False, groups=channels
        )
        self.conv_a = nn.Sequential(
            nn.Conv2d(channels * 2, 32, kernel_size=1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=1, bias=False),
        )
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):
        _, c_lrx, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()
        if c_lrx != self.channels:
            raise RuntimeError(f"Expected {self.channels} input channels, got {c_lrx}")

        n = self.box_filter(x_lr.new_ones((1, self.channels, h_lrx, w_lrx)))
        mean_x = self.box_filter(x_lr) / n
        mean_y = self.box_filter(y_lr) / n
        cov_xy = self.box_filter(x_lr * y_lr) / n - mean_x * mean_y
        var_x = self.box_filter(x_lr * x_lr) / n - mean_x * mean_x

        a = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        b = mean_y - a * mean_x

        mean_a = F.interpolate(a, (h_hrx, w_hrx), mode="bilinear", align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode="bilinear", align_corners=True)
        return mean_a * x_hr + mean_b


def build_lr_net(norm=AdaptiveNorm, layer=5, channels=CHANNELS):
    layers = [
        nn.Conv2d(channels, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    for l in range(1, layer):
        layers += [
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=2**l, dilation=2**l, bias=False),
            norm(24),
            nn.LeakyReLU(0.2, inplace=True),
        ]
    layers += [
        nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(24, channels, kernel_size=1, stride=1, padding=0, dilation=1),
    ]
    net = nn.Sequential(*layers)
    net.apply(weights_init_identity)
    return net


class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=1, channels=CHANNELS):
        super().__init__()
        self.lr = build_lr_net(channels=channels)
        self.gf = ConvGuidedFilter(radius, channels=channels, norm=AdaptiveNorm)

    def forward(self, x_lr, x_hr):
        return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)


def read_rgb01(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32") / 255.0


def rotate_landscape(img):
    return np.ascontiguousarray(np.rot90(img, k=-1))


def crop_to_even(tensor):
    _, _, h, w = tensor.shape
    return tensor[:, :, : h - (h % 2), : w - (w % 2)]


def restore_orientation(image, was_rotated):
    if was_rotated:
        image = np.rot90(image, k=1)
    return np.ascontiguousarray(image)


def save_prediction(prediction, output_path, was_rotated, overwrite):
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return False

    image = prediction.detach().cpu().permute(1, 2, 0).numpy()
    image = restore_orientation(image, was_rotated)
    image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    params = []
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    if not cv2.imwrite(str(output_path), image, params):
        raise RuntimeError(f"Failed to write prediction: {output_path}")
    return True


def resolve_data_dirs(data_root, split):
    data_root = Path(data_root)
    return (
        data_root / f"{split}input_downsample",
        data_root / f"{split}input",
        data_root / f"{split}gt",
    )


class BSDSDataset(Dataset):
    def __init__(self, data_root, split):
        self.x_lr_dir, self.x_hr_dir, self.label_dir = resolve_data_dirs(data_root, split)

        names = self._names(self.x_lr_dir) & self._names(self.x_hr_dir) & self._names(self.label_dir)
        self.names = sorted(names)
        if not self.names:
            raise RuntimeError(f"No matching {split} files found under {data_root}")

    @staticmethod
    def _names(folder):
        return {p.name for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        x_lr = read_rgb01(self.x_lr_dir / name)
        x_hr = read_rgb01(self.x_hr_dir / name)
        label = read_rgb01(self.label_dir / name)
        was_rotated = x_hr.shape[0] > x_hr.shape[1]
        if was_rotated:
            x_lr = rotate_landscape(x_lr)
            x_hr = rotate_landscape(x_hr)
            label = rotate_landscape(label)

        return (
            torch.from_numpy(x_lr).permute(2, 0, 1),
            torch.from_numpy(x_hr).permute(2, 0, 1),
            torch.from_numpy(label).permute(2, 0, 1),
            name,
            was_rotated,
        )


def collate_same_size(batch):
    lr_shapes = {item[0].shape for item in batch}
    hr_shapes = {item[1].shape for item in batch}
    label_shapes = {item[2].shape for item in batch}
    if len(lr_shapes) != 1 or len(hr_shapes) != 1 or len(label_shapes) != 1:
        raise RuntimeError(
            f"Mixed tensor sizes in one batch: LR={sorted(lr_shapes)}, "
            f"HR={sorted(hr_shapes)}, label={sorted(label_shapes)}"
        )
    x_lr, x_hr, label, names, was_rotated = zip(*batch)
    return (
        torch.stack(x_lr, dim=0),
        torch.stack(x_hr, dim=0),
        torch.stack(label, dim=0),
        list(names),
        list(was_rotated),
    )


def load_model_weights(model, model_path, device):
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def calc_psnr_per_image(pred, target, eps=1e-12):
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=eps))


def gaussian_kernel(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    return kernel_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def calc_ssim_per_image(pred, target, window_size=11, sigma=1.5):
    channels = pred.shape[1]
    kernel = gaussian_kernel(window_size, sigma, channels, pred.device, pred.dtype)
    padding = window_size // 2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.conv2d(pred, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=channels)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, kernel, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=padding, groups=channels) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return ssim_map.mean(dim=(1, 2, 3))


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(model, loader, device, output_dir, overwrite=False, warmup_batches=1):
    psnr_sum = 0.0
    ssim_sum = 0.0
    mse_sum = 0.0
    sample_count = 0
    saved_count = 0
    skipped_count = 0
    inference_seconds = 0.0

    model.eval()
    with torch.no_grad():
        for batch_idx, (x_lr, x_hr, label, names, was_rotated) in enumerate(
            tqdm(loader, desc="Evaluating", dynamic_ncols=True)
        ):
            x_hr = crop_to_even(x_hr)
            label = crop_to_even(label)
            x_lr = x_lr.to(device, non_blocking=True) - 0.5
            x_hr = x_hr.to(device, non_blocking=True) - 0.5
            label = label.to(device, non_blocking=True)

            if batch_idx < warmup_batches:
                _ = model(x_lr, x_hr)
                synchronize_if_cuda(device)

            synchronize_if_cuda(device)
            start = time.perf_counter()
            pred = model(x_lr, x_hr)
            synchronize_if_cuda(device)
            inference_seconds += time.perf_counter() - start

            for image, name, rotated in zip(pred, names, was_rotated):
                if save_prediction(image, Path(output_dir) / name, rotated, overwrite):
                    saved_count += 1
                else:
                    skipped_count += 1

            batch_size = pred.shape[0]
            psnr_sum += calc_psnr_per_image(pred, label).sum().item()
            ssim_sum += calc_ssim_per_image(pred, label).sum().item()
            mse_sum += torch.mean((pred - label) ** 2, dim=(1, 2, 3)).sum().item()
            sample_count += batch_size

    return {
        "samples": sample_count,
        "saved": saved_count,
        "skipped": skipped_count,
        "psnr": psnr_sum / sample_count,
        "ssim": ssim_sum / sample_count,
        "mse": mse_sum / sample_count,
        "inference_seconds": inference_seconds,
        "seconds_per_image": inference_seconds / sample_count,
        "fps": sample_count / inference_seconds if inference_seconds > 0 else float("inf"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ConvGuidedFilter denoiser and save BSDS predictions.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Prediction directory (default: <data-root>/refine-<split>input).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup-batches", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    output_dir = args.output_dir or args.data_root / f"refine-{args.split}input"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = BSDSDataset(args.data_root, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_same_size,
    )

    model = DeepGuidedFilter(radius=1, channels=CHANNELS).to(device)
    load_model_weights(model, args.model_path, device)
    total_params, trainable_params = count_parameters(model)

    metrics = evaluate(
        model,
        loader,
        device,
        output_dir=output_dir,
        overwrite=args.overwrite,
        warmup_batches=args.warmup_batches,
    )

    print("\nEvaluation summary")
    print(f"Data root:        {args.data_root}")
    print(f"Split:            {args.split}")
    print(f"Output dir:       {output_dir}")
    print(f"Model path:       {args.model_path}")
    print(f"Device:           {device}")
    print(f"Samples:          {metrics['samples']}")
    print(f"Saved:            {metrics['saved']}")
    print(f"Skipped existing: {metrics['skipped']}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Parameters:       {total_params:,} total / {trainable_params:,} trainable")
    print(f"MSE:              {metrics['mse']:.8f}")
    print(f"PSNR:             {metrics['psnr']:.4f} dB")
    print(f"SSIM:             {metrics['ssim']:.6f}")
    print(f"Inference time:   {metrics['inference_seconds']:.4f} s total")
    print(f"Time per image:   {metrics['seconds_per_image'] * 1000.0:.4f} ms")
    print(f"Throughput:       {metrics['fps']:.2f} images/s")


if __name__ == "__main__":
    main()
