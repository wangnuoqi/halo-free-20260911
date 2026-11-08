"""Exp B: original edge-preserving pipeline with VIV-inspired LR blocks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


PHASES = ("base", "refine", "joint")


def validate_phase(phase: str) -> str:
    phase = str(phase).lower()
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}; got {phase!r}")
    return phase


class AdaptiveNorm(nn.Module):
    """The original adaptive blend of an identity path and BatchNorm."""

    def __init__(self, channels: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.ones(1))
        self.w1 = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(channels, momentum=0.999, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w0 * x + self.w1 * self.bn(x)


def identity_compatible_init(module: nn.Module) -> None:
    """Match the original LRNet's identity initialization where shapes allow."""
    if isinstance(module, nn.Conv2d):
        with torch.no_grad():
            module.weight.zero_()
            out_channels, in_channels, kh, kw = module.weight.shape
            if out_channels < in_channels:
                nn.init.xavier_uniform_(module.weight)
            else:
                module.weight[:, :, kh // 2, kw // 2].diagonal().fill_(1.0)
            # Preserve the source network's Conv2d default bias initialization.
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class VIVResidualBlock(nn.Module):
    """A small VIV-inspired block with an identity-compatible start.

    It combines local and wider depthwise branches, feature-correlation
    weighting, and a factorized fourth-order multiplicative interaction. A
    zero-initialized projection makes the initial block equal to the original
    identity-initialized Conv + AdaptiveNorm + LeakyReLU transform.
    """

    def __init__(self, channels: int = 24, dilation: int = 1):
        super().__init__()
        hidden = channels * 2
        self.channels = channels
        self.dilation = dilation
        self.expand = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.local = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation,
            dilation=dilation, groups=channels, bias=True,
        )
        self.wide = nn.Conv2d(
            channels, channels, kernel_size=9, padding=4,
            groups=channels, bias=True,
        )
        self.order3 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation,
            dilation=dilation, groups=channels, bias=True,
        )
        self.order4 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation,
            dilation=dilation, groups=channels, bias=True,
        )
        mid = max(4, channels // 2)
        self.similarity = nn.Sequential(
            nn.Linear(hidden, mid),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(mid, hidden),
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.norm = AdaptiveNorm(channels)
        self.activation = nn.LeakyReLU(0.2, inplace=False)

        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f3, f9 = self.expand(x).chunk(2, dim=1)
        f3 = self.local(f3)
        f9 = self.wide(f9)

        # Channel correlation is calculated over spatial samples, so cost grows
        # with channel width rather than with the square of image resolution.
        features = torch.cat((f3, f9), dim=1).flatten(2)
        features = F.normalize(features, dim=2, eps=1e-6)
        correlation = torch.bmm(features, features.transpose(1, 2))
        # Reduce pairwise channel correlations to one descriptor per channel
        # before the channel MLP. Feeding the full BxCxC matrix to the MLP
        # would retain an extra channel axis and produce non-broadcastable
        # weights.
        descriptor = correlation.mean(dim=2)
        weights = self.similarity(descriptor)
        weights = torch.tanh(weights).unsqueeze(-1).unsqueeze(-1)
        w3, w9 = weights.chunk(2, dim=1)
        f3 = f3 * (1.0 + w3)
        f9 = f9 * (1.0 + w9)

        # Factorized multiplicative interaction: second, third, then fourth order.
        mixed = f3 * f9
        mixed = self.order3(mixed) * f9
        mixed = self.order4(mixed) * f3
        return self.activation(self.norm(x + self.project(mixed)))


class VIVCompatibleLR(nn.Module):
    """Original LRNet macro-layout with only its four dilated blocks replaced."""

    output_mode = "raw"

    def __init__(self, channels: int = 24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            AdaptiveNorm(channels),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.dilated_blocks = nn.Sequential(
            *(VIVResidualBlock(channels, dilation=2**level) for level in range(1, 5))
        )
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            AdaptiveNorm(channels),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.stem.apply(identity_compatible_init)
        # The internal VIV residual projections are zero initialized by design.
        self.tail.apply(identity_compatible_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.dilated_blocks(self.stem(x)))


class ConvGuidedFilter(nn.Module):
    """Original learned local affine guided-filter operator."""

    def __init__(self):
        super().__init__()
        self.box = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        nn.init.ones_(self.box.weight)
        self.a = nn.Sequential(
            nn.Conv2d(2, 32, 1, bias=False),
            AdaptiveNorm(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 1, bias=False),
            AdaptiveNorm(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 1, 1, bias=False),
        )

    def forward(self, x_lr: torch.Tensor, y_lr: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        count = self.box(torch.ones_like(x_lr))
        mean_x = self.box(x_lr) / count
        mean_y = self.box(y_lr) / count
        covariance = self.box(x_lr * y_lr) / count - mean_x * mean_y
        variance = self.box(x_lr * x_lr) / count - mean_x * mean_x
        a = self.a(torch.cat((covariance, variance), dim=1))
        b = mean_y - a * mean_x
        a_hr = F.interpolate(a, guide.shape[-2:], mode="bilinear", align_corners=True)
        b_hr = F.interpolate(b, guide.shape[-2:], mode="bilinear", align_corners=True)
        return (a_hr * guide + b_hr).clamp(0.0, 1.0)


class DeepGuidedBase(nn.Module):
    """VIV-compatible LR predictor followed by the original ConvGuided filter."""

    def __init__(self):
        super().__init__()
        self.lr = VIVCompatibleLR()
        self.gf = ConvGuidedFilter()

    def forward(self, lr_centered: torch.Tensor, hr_centered: torch.Tensor) -> torch.Tensor:
        return self.gf(lr_centered, self.lr(lr_centered), hr_centered)


def sobel(x: torch.Tensor, eps: float = 1e-6):
    kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + eps)
    return mag, gx, gy


def normalized_sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    mag, _, _ = sobel(x)
    return mag / (mag.amax(dim=(2, 3), keepdim=True) + 1e-6)


class OriginalStage(nn.Module):
    """The original per-pixel stage predictor from the exp7C cascade."""

    def __init__(self, out_channels: int, base_channels: int = 32):
        super().__init__()
        in_channels = 3 if out_channels == 2 else 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_channels, out_channels, 1),
        )

    def forward(self, x, coarse, delta=None):
        grad = normalized_sobel_magnitude(coarse)
        parts = [x, coarse, grad]
        if delta is not None:
            parts.append(delta)
        return self.net(torch.cat(parts, dim=1))


class ThreeStageRefine(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = OriginalStage(2)
        self.s2 = OriginalStage(3)
        self.s3 = OriginalStage(3)

    def forward(self, x: torch.Tensor, coarse0_centered: torch.Tensor):
        q1 = self.s1(x, coarse0_centered)
        d1 = q1[:, :1] * torch.sigmoid(q1[:, 1:2])
        c1 = coarse0_centered + d1

        q2 = self.s2(x, c1, d1)
        a2 = 1.0 + 0.2 * torch.tanh(q2[:, :1])
        b2 = 0.2 * torch.tanh(q2[:, 1:2])
        g2 = torch.sigmoid(q2[:, 2:3])
        d2 = d1 + g2 * (a2 * d1 + b2 - d1)
        c2 = coarse0_centered + d2

        q3 = self.s3(x, c2, d2)
        a3 = 1.0 + 0.2 * torch.tanh(q3[:, :1])
        b3 = 0.2 * torch.tanh(q3[:, 1:2])
        g3 = torch.sigmoid(q3[:, 2:3])
        d3 = d2 + g3 * (a3 * d2 + b3 - d2)
        final = coarse0_centered + d3
        return {
            "coarse1": c1,
            "coarse2": c2,
            "final": final,
            "delta1": d1,
            "delta2": d2,
            "delta3": d3,
        }


class FullExpB(nn.Module):
    """Base plus original three-stage refine, using centered LR/HR inputs."""

    def __init__(self):
        super().__init__()
        self.base = DeepGuidedBase()
        self.refine = ThreeStageRefine()

    def forward(self, lr_centered: torch.Tensor, hr_centered: torch.Tensor):
        coarse0 = self.base(lr_centered, hr_centered)
        refined = self.refine(hr_centered, coarse0 - 0.5)
        result = {
            "coarse0": coarse0,
            "coarse1": (refined["coarse1"] + 0.5).clamp(0.0, 1.0),
            "coarse2": (refined["coarse2"] + 0.5).clamp(0.0, 1.0),
            "final": (refined["final"] + 0.5).clamp(0.0, 1.0),
        }
        result.update({key: refined[key] for key in ("delta1", "delta2", "delta3")})
        return result
