import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import torch.nn.functional as F


import os, datetime
from tqdm import tqdm
import cv2
import numpy as np
import time
import math
import imageio

class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))
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
        center_h, center_w = height // 2, width // 2
        for index in range(n_in):
            module.weight.data[index, index, center_h, center_w] = 1.0
    elif "BatchNorm2d" in name:
        nn.init.constant_(module.weight.data, 1.0)
        nn.init.constant_(module.bias.data, 0.0)


class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1, norm=AdaptiveNorm):
        super().__init__()
        self.box_filter = nn.Conv2d(
            1, 1, kernel_size=3, padding=radius, dilation=radius,
            bias=False, groups=1,
        )
        self.conv_a = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
        )
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):
        _, _, h_lr, w_lr = x_lr.size()
        _, _, h_hr, w_hr = x_hr.size()
        count = self.box_filter(x_lr.new_ones((1, 1, h_lr, w_lr)))
        mean_x = self.box_filter(x_lr) / count
        mean_y = self.box_filter(y_lr) / count
        cov_xy = self.box_filter(x_lr * y_lr) / count - mean_x * mean_y
        var_x = self.box_filter(x_lr * x_lr) / count - mean_x * mean_x
        a = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        b = mean_y - a * mean_x
        mean_a = F.interpolate(a, (h_hr, w_hr), mode="bilinear", align_corners=True)
        mean_b = F.interpolate(b, (h_hr, w_hr), mode="bilinear", align_corners=True)
        return mean_a * x_hr + mean_b


def build_lr_net(norm=AdaptiveNorm, layer=5):
    layers = [
        nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    for level in range(1, layer):
        layers.extend([
            nn.Conv2d(24, 24, kernel_size=3, padding=2 ** level,
                      dilation=2 ** level, bias=False),
            norm(24),
            nn.LeakyReLU(0.2, inplace=True),
        ])
    layers.extend([
        nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(24, 1, kernel_size=1),
    ])
    network = nn.Sequential(*layers)
    network.apply(weights_init_identity)
    return network


class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8):
        super().__init__()
        self.lr = build_lr_net()
        self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm)

    def forward(self, x_lr, x_hr):
        return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)


def load_frozen_stage1(model_path, device):
    stage1_model = DeepGuidedFilter(radius=1, eps=1e-8).to(device)
    state_dict = torch.load(model_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    stage1_model.load_state_dict(state_dict, strict=True)
    stage1_model.eval()
    for parameter in stage1_model.parameters():
        parameter.requires_grad_(False)
    return stage1_model

class Stage1DeltaNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, 2, kernel_size=1, bias=True)  # raw delta, raw edge gate
        )

    def forward(self, x_raw, x_coarse):
        g, _, _ = compute_sobel_grad(x_coarse, normalize=True)
        inp = torch.cat([x_raw, x_coarse, g], dim=1)
        raw = self.net(inp)
        delta_raw = raw[:, 0:1]
        gate = torch.sigmoid(raw[:, 1:2])
        return delta_raw * gate, gate

class ABAdjustNet(nn.Module):
    def __init__(self, base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True):
        super().__init__()
        self.a_scale = a_scale
        self.b_scale = b_scale
        self.use_delta_as_input = use_delta_as_input

        in_ch = 3 + (1 if use_delta_as_input else 0)  # raw + coarse_k + grad + (delta_prev可选)

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, 3, 1, bias=True)  # (a_raw, b_raw, gate_raw) each pixel
        )

    def forward(self, x_raw, x_coarse_k, delta_prev=None):
        """
        x_coarse_k: 每一轮“更新后的 coarse”，来自上一轮 coarse + refine 的和（不是 concat）
        a,b: per-pixel (B,1,H,W)
        """
        g, _, _ = compute_sobel_grad(x_coarse_k, normalize=True)

        if self.use_delta_as_input:
            assert delta_prev is not None
            inp = torch.cat([x_raw, x_coarse_k, g, delta_prev], dim=1)
        else:
            inp = torch.cat([x_raw, x_coarse_k, g], dim=1)

        ab = self.net(inp)
        a_raw = ab[:, 0:1]
        b_raw = ab[:, 1:2]
        gate = torch.sigmoid(ab[:, 2:3])

        # per-pixel a,b（关键：这里就是每个像素一个 a、一个 b）
        a = 1.0 + self.a_scale * torch.tanh(a_raw)
        b = self.b_scale * torch.tanh(b_raw)
        return a, b, gate

class CascadeRefine3Stage(nn.Module):
    def __init__(self, base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True):
        super().__init__()
        self.stage1 = Stage1DeltaNet(base_channels=base_channels)
        self.stage2 = ABAdjustNet(base_channels=base_channels, a_scale=a_scale, b_scale=b_scale,
                                 use_delta_as_input=use_delta_as_input)
        self.stage3 = ABAdjustNet(base_channels=base_channels, a_scale=a_scale, b_scale=b_scale,
                                 use_delta_as_input=use_delta_as_input)

    def forward(self, x_raw, x_coarse0):
        # ---- Stage 1 ----
        delta1, gate1 = self.stage1(x_raw, x_coarse0)     # gated refine delta
        x_coarse1 = x_coarse0 + delta1                    # 更新 coarse（相加）
        # 这里 x_coarse1 就是你说的 “refine output + coarse input 的和”

        # ---- Stage 2 ----
        a2, b2, gate2 = self.stage2(x_raw, x_coarse1, delta1)  # per-pixel a,b,gate
        proposed_delta2 = a2 * delta1 + b2
        delta2 = delta1 + gate2 * (proposed_delta2 - delta1)
        x_coarse2 = x_coarse0 + delta2                    # 更新 coarse（相加得到下一轮 coarse）

        # ---- Stage 3 ----
        a3, b3, gate3 = self.stage3(x_raw, x_coarse2, delta2)
        proposed_delta3 = a3 * delta2 + b3
        delta3 = delta2 + gate3 * (proposed_delta3 - delta2)
        x_final = x_coarse0 + delta3                      # 最终输出图

        return {
            "delta1": delta1, "coarse1": x_coarse1,
            "delta2": delta2, "coarse2": x_coarse2,
            "delta3": delta3, "final":  x_final,
            "a2": a2, "b2": b2, "gate1": gate1, "gate2": gate2,
            "a3": a3, "b3": b3, "gate3": gate3,
        }

def cascade_loss(outputs, x_coarse0, gt,
                 thresh_mode='otsu', thresh=0.6, dilate=0,
                 loss_type='l1', normalize_grad=True):
    delta_gt = gt - x_coarse0

    l1 = edge_guided_loss(outputs["delta1"],  delta_gt,
                          thresh_mode=thresh_mode, thresh=thresh, dilate=dilate,
                          loss_type=loss_type, normalize_grad=normalize_grad)
    l2 = edge_guided_loss(outputs["delta2"], delta_gt,
                          thresh_mode=thresh_mode, thresh=thresh, dilate=dilate,
                          loss_type=loss_type, normalize_grad=normalize_grad)
    l3 = edge_guided_loss(outputs["delta3"], delta_gt,
                          thresh_mode=thresh_mode, thresh=thresh, dilate=dilate,
                          loss_type=loss_type, normalize_grad=normalize_grad)

    return 0.1*l1 + 0.3*l2 + 0.6*l3, (l1, l2, l3)

# --------- 工具：Sobel 梯度幅值（对灰度图）---------
def compute_sobel_grad(x, eps=1e-6, normalize=False):
    # x: Bx1xHxW (若不是，会自动做灰度)
    #x = to_gray(x)
    sobel_x = torch.tensor([[[[-1., 0., 1.],
                              [-2., 0., 2.],
                              [-1., 0., 1.]]]], device=x.device, dtype=x.dtype)
    sobel_y = torch.tensor([[[[-1., -2., -1.],
                              [ 0.,  0.,  0.],
                              [ 1.,  2.,  1.]]]], device=x.device, dtype=x.dtype)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    g  = torch.sqrt(gx*gx + gy*gy + eps)
    if normalize:
        g = g / (g.amax(dim=(2,3), keepdim=True) + eps)
    return g, gx, gy  # Bx1xHxW

def batched_otsu_threshold(g, num_bins=256, eps=1e-6, min_delta=1e-6):
    """
    g: (B,1,H,W) 梯度幅值图
    返回: (B,) 每张图的 Otsu 阈值（与 g 同物理量纲）
    """
    assert g.dim() == 4 and g.shape[1] == 1, "g must be (B,1,H,W)"
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")

    B = g.shape[0]
    flat = g.flatten(1)

    g_min = flat.min(dim=1).values
    g_max = flat.max(dim=1).values
    g_span = g_max - g_min
    span = g_span.clamp_min(eps)

    norm = ((flat - g_min[:, None]) / span[:, None]).clamp(0.0, 1.0)
    bin_idx = (norm * (num_bins - 1)).long()

    offsets = (torch.arange(B, device=g.device) * num_bins)[:, None]
    hist_idx = (bin_idx + offsets).reshape(-1)
    hist = torch.bincount(hist_idx, minlength=B * num_bins).reshape(B, num_bins).to(g.dtype)

    prob = hist / hist.sum(dim=1, keepdim=True).clamp_min(eps)
    centers = torch.linspace(0.0, 1.0, steps=num_bins, device=g.device, dtype=g.dtype).view(1, -1)

    omega = torch.cumsum(prob, dim=1)
    mu = torch.cumsum(prob * centers, dim=1)
    mu_total = mu[:, -1:]

    denom = (omega * (1.0 - omega)).clamp_min(eps)
    sigma_between = ((mu_total * omega - mu) ** 2) / denom
    sigma_between[:, 0] = 0.0
    sigma_between[:, -1] = 0.0

    best_bin = torch.argmax(sigma_between, dim=1)
    t_norm = centers[0, best_bin]
    thresh = g_min + t_norm * span

    # 极小值保护：避免阈值贴到最大值导致无边缘
    valid_span = g_span > min_delta
    high = g_max - min_delta
    thresh = torch.where(valid_span, torch.minimum(thresh, high), g_max)
    return thresh

def per_channel_loss(pred, target, loss_type='l1'):
    if loss_type == 'l1':
        return (pred - target).abs()
    elif loss_type == 'l2':
        d = pred - target
        return d * d
    elif loss_type == 'huber':
        # smooth_l1_loss 返回已reduction的标量或逐像素；这里要逐像素
        return F.smooth_l1_loss(pred, target, reduction='none')
    else:
        raise ValueError("loss_type must be 'l1' | 'l2' | 'huber'")

def masked_reduce(per, mask=None, eps=1e-8, reduction='mean'):
    """
    per: (B,1,H,W) 或 (B,C,H,W) 的逐像素损失
    mask: (B,1,H,W) 的0/1掩膜; None 表示不掩膜
    reduction: 'mean' | 'sum'
    """
    if mask is not None:
        per = per * mask
        if reduction == 'mean':
            denom = mask.sum().clamp_min(eps)
            return per.sum() / denom
        elif reduction == 'sum':
            return per.sum()
        else:
            return per
    else:
        if reduction == 'mean':
            return per.mean()
        elif reduction == 'sum':
            return per.sum()
        else:
            return per

def edge_guided_loss(
    pred, target,
    thresh_mode='otsu',
    thresh=0.9,
    dilate=1,
    loss_type='l1',
    eps=1e-6,
    normalize_grad=True,
    detach_mask=True,
):
    assert pred.shape == target.shape

    # 1) edge mask from GT grad
    g, _, _ = compute_sobel_grad(target, eps=eps, normalize=normalize_grad)  # (B,1,H,W)

    if thresh_mode == 'fixed':
        T = torch.as_tensor(thresh, device=pred.device, dtype=pred.dtype)
        edge_mask = (g > T).float()
    elif thresh_mode == 'percentile':
        q = torch.quantile(g.flatten(1), torch.as_tensor(thresh, device=g.device), dim=1)
        edge_mask = (g > q.view(-1,1,1,1)).float()
    elif thresh_mode == 'otsu':
        T = batched_otsu_threshold(g, num_bins=256, eps=eps, min_delta=eps)
        edge_mask = (g > T.view(-1,1,1,1)).float()

        # 无边缘保护：当某张图没有边缘像素时，保留其最大梯度像素为边缘
        edge_count = edge_mask.flatten(1).sum(dim=1)
        no_edge = edge_count <= 0
        if no_edge.any():
            flat_g = g.flatten(1)
            max_idx = flat_g.argmax(dim=1, keepdim=True)
            fallback = torch.zeros_like(flat_g)
            fallback.scatter_(1, max_idx, 1.0)
            flat_mask = edge_mask.flatten(1)
            flat_mask[no_edge] = fallback[no_edge]
            edge_mask = flat_mask.view_as(edge_mask)
    else:
        raise ValueError("thresh_mode must be 'fixed' | 'percentile' | 'otsu'")

    if dilate and dilate > 0:
        k = dilate * 2 + 1
        edge_mask = F.max_pool2d(edge_mask, kernel_size=k, stride=1, padding=dilate)

    if detach_mask:
        edge_mask = edge_mask.detach()

    smooth_mask = (1.0 - edge_mask).clamp(0.0, 1.0)

    # 2) pixel loss map (mean over C to avoid channel-count scaling)
    per_pix = per_channel_loss(pred, target, loss_type=loss_type)  # (B,C,H,W) or (B,1,H,W)
    if per_pix.dim() == 4 and per_pix.shape[1] > 1:
        per_pix = per_pix.mean(dim=1, keepdim=True)                # -> (B,1,H,W)

    edge_pix = masked_reduce(per_pix, edge_mask,  eps=eps, reduction='mean')
    smooth_pix = masked_reduce(per_pix, smooth_mask, eps=eps, reduction='mean')

    # 3) grad loss (only on edge)
    _, gpx, gpy = compute_sobel_grad(pred,   eps=eps, normalize=normalize_grad)
    _, gtx, gty = compute_sobel_grad(target, eps=eps, normalize=normalize_grad)
    grad_diff = (gpx - gtx).abs() + (gpy - gty).abs()
    edge_grad = masked_reduce(grad_diff, edge_mask, eps=eps, reduction='mean')

    # exp1: keep the original edge/smooth split, but use explicit weights.
    # The previous dynamic ratios forced all three terms to the same scale,
    # which effectively removed control over the gradient and smooth losses.
    total = (
        1.0 * smooth_pix
        + 2.0 * edge_pix
        + 0.2 * edge_grad
    )
    return total


# -----------------------------------------------------------------------------
# Clip-art data adapter. The model/loss definitions above are copied from
# exp7C_joint_fidelity_20260818.py. Only the input/label data path is changed.
# -----------------------------------------------------------------------------
import argparse
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "openclipart_5000" / "jpeg_pairs_512"
DEFAULT_CONV_CHECKPOINT = (
    PROJECT_ROOT.parent / "20260815_edgepreserving" / "models" /
    "convguided_20260818_l0p500_a1p400" /
    "convguided_filter_best_20260818_l0p500_a1p400.pth"
)


def parse_args():
    p = argparse.ArgumentParser(description="Train exp7C-style joint model on clip-art JPEG pairs")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                   help="directory containing clean_png and jpeg_qXX")
    p.add_argument("--quality", type=int, default=30, choices=[10, 20, 30, 50])
    p.add_argument("--conv-checkpoint", type=Path, default=DEFAULT_CONV_CHECKPOINT)
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "checkpoints" / "clipart_joint_q30")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--list-dir", type=Path, default=None,
                   help="directory containing train_list.txt, val_list.txt, test_list.txt; defaults to data-root parent")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def y_channel(path: Path):
    with Image.open(path) as image:
        ycbcr = image.convert("YCbCr")
        y = ycbcr.getchannel(0)
        return np.asarray(y, dtype=np.float32) / 255.0


def half_resolution(y):
    h, w = y.shape
    target = (max(1, w // 2), max(1, h // 2))
    image = Image.fromarray(np.clip(y * 255.0, 0, 255).astype(np.uint8), mode="L")
    return np.asarray(image.resize(target, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0


class ClipartJpegDataset(Dataset):
    """JPEG Y input + its clean PNG Y target; LR is derived from degraded Y."""
    def __init__(self, data_root: Path, quality: int, split: str, list_dir: Path = None):
        self.data_root = data_root
        self.clean_dir = data_root / "clean_png"
        self.jpeg_dir = data_root / f"jpeg_q{quality:02d}"
        if not self.clean_dir.is_dir() or not self.jpeg_dir.is_dir():
            raise FileNotFoundError(f"Expected {self.clean_dir} and {self.jpeg_dir}")
        list_path = (list_dir or data_root.parent) / f"{split}_list.txt"
        stems = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if not stems or len(stems) != len(set(stems)):
            raise ValueError(f"Empty list or duplicate entries: {list_path}")
        self.pairs = []
        for stem in stems:
            if Path(stem).name != stem or Path(stem).suffix in {".jpg", ".png"}:
                raise ValueError(f"Expected image stem without extension: {stem}")
            jpeg = self.jpeg_dir / f"{stem}.jpg"
            clean = self.clean_dir / f"{stem}.png"
            if not jpeg.is_file() or not clean.is_file():
                raise FileNotFoundError(f"Missing pair listed in {list_path}: {stem}")
            self.pairs.append((jpeg, clean))
        self.split = split
        print(f"{split}: {len(self.pairs)} pairs (quality={quality}, list={list_path})")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        jpeg_path, clean_path = self.pairs[idx]
        y_hr = y_channel(jpeg_path)
        y_lr = half_resolution(y_hr)
        y_gt = y_channel(clean_path)
        return (torch.from_numpy(y_lr[None]),
                torch.from_numpy(y_hr[None]),
                torch.from_numpy(y_gt[None]))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_set = ClipartJpegDataset(args.data_root, args.quality, "train", args.list_dir)
    val_set = ClipartJpegDataset(args.data_root, args.quality, "val", args.list_dir)
    test_set = ClipartJpegDataset(args.data_root, args.quality, "test", args.list_dir)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                              drop_last=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers)

    stage1_model = DeepGuidedFilter(radius=1, eps=1e-8).to(device)
    state = torch.load(args.conv_checkpoint, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    stage1_model.load_state_dict(state, strict=True)
    for parameter in stage1_model.parameters():
        parameter.requires_grad_(True)

    model = CascadeRefine3Stage(base_channels=32, a_scale=0.2, b_scale=0.2,
                                use_delta_as_input=True).to(device)
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": 3e-3},
        {"params": stage1_model.parameters(), "lr": 3e-4},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    writer = SummaryWriter(str(args.output_dir / "tensorboard"))
    best_mse = float("inf")

    def evaluate(loader):
        model.eval(); stage1_model.eval(); total_mse = 0.0; total_psnr = 0.0
        with torch.no_grad():
            for x_lr, x_hr, gt in loader:
                x_lr, x_hr, gt = x_lr.to(device), x_hr.to(device), gt.to(device)
                coarse = stage1_model(x_lr - 0.5, x_hr - 0.5)
                output = model(x_hr - 0.5, coarse - 0.5)["final"] + 0.5
                output = torch.clamp(output, 0, 1)
                mse = torch.mean((output - gt) ** 2).item()
                total_mse += mse
                total_psnr += (100.0 if mse == 0 else 10 * np.log10(1.0 / mse))
        return total_mse / len(loader), total_psnr / len(loader)

    for epoch in range(args.epochs):
        model.train(); stage1_model.train()
        running = 0.0
        for batch_idx, (x_lr, x_hr, gt) in enumerate(train_loader):
            x_lr, x_hr, gt = x_lr.to(device), x_hr.to(device), gt.to(device)
            coarse = stage1_model(x_lr - 0.5, x_hr - 0.5)
            x_raw = x_hr - 0.5
            coarse0 = coarse - 0.5
            target = gt - 0.5
            outputs = model(x_raw, coarse0)
            loss, (l1, l2, l3) = cascade_loss(
                outputs, coarse0, target, thresh_mode="otsu", dilate=0,
                loss_type="l1", normalize_grad=True)
            loss = loss + 0.5 * F.smooth_l1_loss(coarse0, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        train_loss = running / len(train_loader)
        val_mse, val_psnr = evaluate(val_loader)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/mse", val_mse, epoch)
        writer.add_scalar("val/psnr", val_psnr, epoch)
        print(f"Epoch {epoch+1}/{args.epochs} loss={train_loss:.6f} val_mse={val_mse:.6f} val_psnr={val_psnr:.2f} dB")
        if val_mse < best_mse:
            best_mse = val_mse
            torch.save(model.state_dict(), args.output_dir / "rf_3stage_best.pth")
            torch.save(stage1_model.state_dict(), args.output_dir / "convguided_joint_best.pth")
            (args.output_dir / "best_metrics.txt").write_text(
                f"epoch={epoch+1}\nval_mse={val_mse:.10f}\nval_psnr={val_psnr:.6f}\n",
                encoding="utf-8")
    test_mse, test_psnr = evaluate(test_loader)
    (args.output_dir / "test_metrics.txt").write_text(
        f"test_mse={test_mse:.10f}\ntest_psnr={test_psnr:.6f}\n",
        encoding="utf-8")
    writer.close()
    print(f"Test MSE={test_mse:.6f}, PSNR={test_psnr:.2f} dB")


if __name__ == "__main__":
    main()
