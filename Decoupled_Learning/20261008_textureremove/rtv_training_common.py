"""Shared model, dataset, and loss code for the RTV texture-removal runs."""

from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AdaptiveNorm(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.w_0 = nn.Parameter(torch.tensor([1.0]))
        self.w_1 = nn.Parameter(torch.tensor([0.0]))
        self.bn = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


def weights_init_identity(module):
    name = module.__class__.__name__
    if "Conv" in name:
        n_out, n_in, height, width = module.weight.data.size()
        if n_out < n_in:
            nn.init.xavier_uniform_(module.weight.data)
            return
        module.weight.data.zero_()
        for index in range(n_in):
            module.weight.data[index, index, height // 2, width // 2] = 1.0
    elif "BatchNorm2d" in name:
        nn.init.constant_(module.weight.data, 1.0)
        nn.init.constant_(module.bias.data, 0.0)


class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1, norm=AdaptiveNorm, channels=3):
        super().__init__()
        self.channels = channels
        self.box_filter = nn.Conv2d(
            channels, channels, kernel_size=3, padding=radius, dilation=radius,
            bias=False, groups=channels,
        )
        self.conv_a = nn.Sequential(
            nn.Conv2d(2 * channels, 32, kernel_size=1, bias=False),
            norm(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            norm(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=1, bias=False),
        )
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):
        _, _, h_lr, w_lr = x_lr.size()
        _, _, h_hr, w_hr = x_hr.size()
        count = self.box_filter(x_lr.new_ones((1, self.channels, h_lr, w_lr)))
        mean_x = self.box_filter(x_lr) / count
        mean_y = self.box_filter(y_lr) / count
        cov_xy = self.box_filter(x_lr * y_lr) / count - mean_x * mean_y
        var_x = self.box_filter(x_lr * x_lr) / count - mean_x * mean_x
        a = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        b = mean_y - a * mean_x
        mean_a = F.interpolate(a, (h_hr, w_hr), mode="bilinear", align_corners=True)
        mean_b = F.interpolate(b, (h_hr, w_hr), mode="bilinear", align_corners=True)
        return mean_a * x_hr + mean_b


def build_lr_net(norm=AdaptiveNorm, layer=5, in_channels=3, out_channels=3):
    layers = [
        nn.Conv2d(in_channels, 24, kernel_size=3, padding=1, bias=False),
        norm(24), nn.LeakyReLU(0.2, inplace=True),
    ]
    for level in range(1, layer):
        layers += [
            nn.Conv2d(24, 24, kernel_size=3, padding=2 ** level,
                      dilation=2 ** level, bias=False),
            norm(24), nn.LeakyReLU(0.2, inplace=True),
        ]
    layers += [
        nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False),
        norm(24), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(24, out_channels, kernel_size=1),
    ]
    network = nn.Sequential(*layers)
    network.apply(weights_init_identity)
    return network


class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8, channels=3):
        super().__init__()
        self.channels = channels
        self.lr = build_lr_net(in_channels=channels, out_channels=channels)
        self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm, channels=channels)

    def forward(self, x_lr, x_hr):
        return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)


class Stage1DeltaNet(nn.Module):
    def __init__(self, base_channels=32, channels=3):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Conv2d(3 * channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, channels, 1, bias=False),
        )

    def forward(self, x_raw, x_coarse):
        g, _, _ = compute_sobel_grad(x_coarse, normalize=True)
        return self.net(torch.cat([x_raw, x_coarse, g], dim=1))


class ABAdjustNet(nn.Module):
    def __init__(self, base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True, channels=3):
        super().__init__()
        self.channels = channels
        self.a_scale = a_scale
        self.b_scale = b_scale
        self.use_delta_as_input = use_delta_as_input
        in_channels = 4 * channels if use_delta_as_input else 3 * channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 2 * channels, 1),
        )

    def forward(self, x_raw, x_coarse, delta_prev=None):
        g, _, _ = compute_sobel_grad(x_coarse, normalize=True)
        values = [x_raw, x_coarse, g]
        if self.use_delta_as_input:
            if delta_prev is None:
                raise ValueError("delta_prev is required when use_delta_as_input=True")
            values.append(delta_prev)
        ab = self.net(torch.cat(values, dim=1))
        c = self.channels
        return 1.0 + self.a_scale * torch.tanh(ab[:, 0:c]), self.b_scale * torch.tanh(ab[:, c:2 * c])


class CascadeRefine3Stage(nn.Module):
    def __init__(self, base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True, channels=3):
        super().__init__()
        self.channels = channels
        self.stage1 = Stage1DeltaNet(base_channels, channels)
        self.stage2 = ABAdjustNet(base_channels, a_scale, b_scale, use_delta_as_input, channels)
        self.stage3 = ABAdjustNet(base_channels, a_scale, b_scale, use_delta_as_input, channels)

    def forward(self, x_raw, x_coarse0):
        delta1 = self.stage1(x_raw, x_coarse0)
        coarse1 = x_coarse0 + delta1
        a2, b2 = self.stage2(x_raw, coarse1, delta1)
        delta2 = a2 * delta1 + b2
        coarse2 = x_coarse0 + delta2
        a3, b3 = self.stage3(x_raw, coarse2, delta2)
        delta3 = a3 * delta2 + b3
        return {
            "delta1": delta1, "coarse1": coarse1,
            "delta2": delta2, "coarse2": coarse2,
            "delta3": delta3, "final": x_coarse0 + delta3,
        }


def _image_files(directory: str):
    extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in extensions)


class RTVDataset(Dataset):
    """Paired ordinary input and fixed-RTV target from one split directory."""
    def __init__(self, split_root: str):
        self.input_files = _image_files(os.path.join(split_root, "input"))
        self.label_files = _image_files(os.path.join(split_root, "label"))
        labels = {p.stem: p for p in self.label_files}
        pairs = [(p, labels[p.stem]) for p in self.input_files if p.stem in labels]
        if not pairs:
            raise RuntimeError(f"No input/label pairs found under {split_root}")
        self.pairs = pairs
        print(f"{split_root}: {len(self.pairs)} input/label pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_path, label_path = self.pairs[index]
        image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if image is None or label is None:
            raise IOError(f"Unable to read pair: {input_path}, {label_path}")
        image = image.astype(np.float32)
        label = label.astype(np.float32)
        if image.ndim == 2:
            image = image[..., None]
        if label.ndim == 2:
            label = label[..., None]
        if image.ndim != 3 or label.ndim != 3 or image.shape[2] != label.shape[2]:
            raise ValueError(
                f"Input/label channel mismatch: {image.shape} vs {label.shape}"
            )
        if not np.isfinite(image).all() or not np.isfinite(label).all():
            raise ValueError(f"Non-finite values found in pair: {input_path}, {label_path}")
        # MATLAB writes float TIFFs in [0, 1]. Keep this guard for files
        # converted by other tools that may store the same values as 8/16 bit.
        if image.max() > 1.0 or label.max() > 1.0:
            image = image / (255.0 if image.max() <= 255.0 else 65535.0)
            label = label / (255.0 if label.max() <= 255.0 else 65535.0)
        image = np.clip(image, 0.0, 1.0)
        label = np.clip(label, 0.0, 1.0)
        h = min(image.shape[0], label.shape[0])
        w = min(image.shape[1], label.shape[1])
        image = np.ascontiguousarray(image[:h, :w].transpose(2, 0, 1))
        label = np.ascontiguousarray(label[:h, :w].transpose(2, 0, 1))
        return torch.from_numpy(image), torch.from_numpy(label)


def collate_pairs(batch):
    h = min(x.shape[-2] for x, _ in batch)
    w = min(x.shape[-1] for x, _ in batch)
    h -= h % 2
    w -= w % 2
    inputs = torch.stack([x[:, :h, :w] for x, _ in batch])
    labels = torch.stack([y[:, :h, :w] for _, y in batch])
    return inputs, labels


def make_low_resolution(image, scale=0.5):
    """Create the ConvGuided low-resolution input from a full-resolution image."""
    height, width = image.shape[-2:]
    low_height = max(2, int(round(height * scale)))
    low_width = max(2, int(round(width * scale)))
    return F.interpolate(image, size=(low_height, low_width), mode="bilinear", align_corners=False)


def compute_sobel_grad(x, eps=1e-6, normalize=False):
    channels = x.size(1)
    kernel_x = [-1., 0., 1., -2., 0., 2., -1., 0., 1.]
    kernel_y = [-1., -2., -1., 0., 0., 0., 1., 2., 1.]
    sobel_x = x.new_tensor(kernel_x).view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
    sobel_y = x.new_tensor(kernel_y).view(1, 1, 3, 3).expand(channels, 1, 3, 3).contiguous()
    gx = F.conv2d(x, sobel_x, padding=1, groups=channels)
    gy = F.conv2d(x, sobel_y, padding=1, groups=channels)
    magnitude = torch.sqrt(gx * gx + gy * gy + eps)
    if normalize:
        magnitude = magnitude / (magnitude.amax(dim=(2, 3), keepdim=True) + eps)
    return magnitude, gx, gy


def base_loss(pred, target):
    pixel = F.mse_loss(pred, target)
    _, px, py = compute_sobel_grad(pred)
    _, tx, ty = compute_sobel_grad(target)
    gradient = (px - tx).abs().mean() + (py - ty).abs().mean()
    gradient_scale = (gradient / (pixel + 1e-8)).detach().clamp_min(1e-8)
    return pixel + gradient / gradient_scale


def _otsu_threshold(gradient, bins=256, eps=1e-6):
    batch = gradient.shape[0]
    flat = gradient.flatten(1)
    minimum = flat.min(dim=1).values
    maximum = flat.max(dim=1).values
    span = (maximum - minimum).clamp_min(eps)
    normalized = ((flat - minimum[:, None]) / span[:, None]).clamp(0.0, 1.0)
    bin_index = (normalized * (bins - 1)).long()
    offsets = (torch.arange(batch, device=gradient.device) * bins)[:, None]
    histogram = torch.bincount((bin_index + offsets).reshape(-1), minlength=batch * bins)
    histogram = histogram.reshape(batch, bins).to(gradient.dtype)
    probability = histogram / histogram.sum(dim=1, keepdim=True).clamp_min(eps)
    centers = torch.linspace(0.0, 1.0, bins, device=gradient.device, dtype=gradient.dtype)[None, :]
    omega = torch.cumsum(probability, dim=1)
    mean = torch.cumsum(probability * centers, dim=1)
    mean_total = mean[:, -1:]
    variance = ((mean_total * omega - mean) ** 2) / (omega * (1.0 - omega)).clamp_min(eps)
    variance[:, 0] = 0.0
    variance[:, -1] = 0.0
    best = torch.argmax(variance, dim=1)
    return minimum + centers[0, best] * span


def _edge_guided_loss(pred, target, dilate=0, eps=1e-6):
    magnitude, _, _ = compute_sobel_grad(target, eps=eps, normalize=True)
    threshold = _otsu_threshold(magnitude, eps=eps)
    edge = (magnitude > threshold[:, None, None, None]).to(pred.dtype)
    if dilate > 0:
        edge = F.max_pool2d(edge, kernel_size=2 * dilate + 1, stride=1, padding=dilate)
    if torch.any(edge.flatten(1).sum(dim=1) == 0):
        empty = edge.flatten(1).sum(dim=1) == 0
        flat = magnitude.flatten(1)
        fallback = torch.zeros_like(flat)
        fallback.scatter_(1, flat.argmax(dim=1, keepdim=True), 1.0)
        edge_flat = edge.flatten(1)
        edge_flat[empty] = fallback[empty]
        edge = edge_flat.view_as(edge)
    smooth = 1.0 - edge
    pixel = (pred - target).abs()
    edge_pixel = (pixel * edge).sum() / edge.sum().clamp_min(eps)
    smooth_pixel = (pixel * smooth).sum() / smooth.sum().clamp_min(eps)
    _, pred_x, pred_y = compute_sobel_grad(pred, eps=eps, normalize=True)
    _, target_x, target_y = compute_sobel_grad(target, eps=eps, normalize=True)
    grad = (pred_x - target_x).abs() + (pred_y - target_y).abs()
    edge_grad = (grad * edge).sum() / edge.sum().clamp_min(eps)
    scale_edge = (edge_grad / (edge_pixel + eps)).detach().clamp_min(eps)
    scale_smooth = (smooth_pixel / (edge_pixel + eps)).detach().clamp_min(eps)
    return edge_pixel + edge_grad / scale_edge + smooth_pixel / scale_smooth


def cascade_loss(outputs, coarse, target):
    delta_target = target - coarse
    losses = [_edge_guided_loss(outputs[key], delta_target) for key in ("delta1", "delta2", "delta3")]
    return sum(losses) / 3.0


def load_state(model, path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    # Checkpoint keys may come from a plain module or DataParallel. Match the
    # keys expected by the model being loaded instead of stripping blindly.
    model_keys = model.state_dict().keys()
    state_has_module = bool(state) and all(k.startswith("module.") for k in state)
    model_has_module = any(k.startswith("module.") for k in model_keys)
    if state_has_module and not model_has_module:
        state = {k[7:]: v for k, v in state.items()}
    elif model_has_module and state and not state_has_module:
        state = {f"module.{k}": v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model
