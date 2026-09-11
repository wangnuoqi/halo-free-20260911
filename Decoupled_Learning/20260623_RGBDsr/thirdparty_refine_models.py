"""Third-party x8 coarse adapters and the two RF variants used in this project."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent


def _pil_bicubic_upsample(depth_lr, size):
    """Upsample normalized depth with the PIL bicubic path used by DKN."""
    if depth_lr.ndim != 4 or depth_lr.shape[1] != 1:
        raise ValueError(f"Expected depth tensor [N,1,H,W], got {tuple(depth_lr.shape)}")
    height, width = size
    source = depth_lr.detach().to(device="cpu", dtype=torch.float32).numpy()
    outputs = []
    for sample in source[:, 0]:
        image = Image.fromarray(np.ascontiguousarray(sample))
        resized = np.asarray(
            image.resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32
        ).copy()
        outputs.append(torch.from_numpy(resized).unsqueeze(0))
    return torch.stack(outputs, dim=0).to(device=depth_lr.device, dtype=depth_lr.dtype)


def _postprocess_coarse(coarse, clamp=True):
    return coarse.clamp(0.0, 1.0) if clamp else coarse


def _prepare_dkn_rgb(rgb, bgr=True):
    return rgb[:, [2, 1, 0]] if bgr else rgb


def _prepare_dctnet_rgb(rgb):
    mean = rgb.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = rgb.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (rgb - mean) / std


def _strip_module(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and state and all(k.startswith("module.") for k in state):
        return {k[7:]: v for k, v in state.items()}
    return state


class ThirdPartyCoarse(nn.Module):
    """Load one official x8 third-party model and expose (depth_lr, rgb)->HR coarse."""

    def __init__(self, method: str, checkpoint: Path, device: torch.device,
                 clamp_output=True, bgr_input=True):
        super().__init__()
        method = method.lower()
        if method not in self.supported_methods():
            raise ValueError(f"Unsupported third-party model: {method}")
        self.method = method
        self.clamp_output = bool(clamp_output)
        self.bgr_input = bool(bgr_input)
        third_party = PROJECT_ROOT / "third_party" / ("DKN" if method == "dkn" else "DCTNet")
        sys.path.insert(0, str(third_party))
        try:
            if method == "dkn":
                from models import DKN
                self.net = DKN(kernel_size=3, filter_size=15, residual=True).to(device)
            else:
                from model import DCTNet
                self.net = DCTNet().to(device)
        finally:
            sys.path.pop(0)
        state = _strip_module(torch.load(checkpoint, map_location=device))
        self.net.load_state_dict(state, strict=True)
        self.net.eval()
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def supported_methods():
        return ("dkn", "dctnet")

    def forward(self, depth_lr, rgb):
        height, width = rgb.shape[-2:]
        depth_hr = _pil_bicubic_upsample(depth_lr, (height, width))
        if self.method == "dkn":
            # The released DKN implementation consumes OpenCV/BGR channel order.
            coarse = self.net((_prepare_dkn_rgb(rgb, bgr=self.bgr_input), depth_hr))
        else:
            coarse = self.net(depth_hr, _prepare_dctnet_rgb(rgb))
        return _postprocess_coarse(coarse, clamp=self.clamp_output)


def _sobel(x):
    kx = x.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
    gx, gy = F.conv2d(x, kx, padding=1), F.conv2d(x, ky, padding=1)
    magnitude = torch.sqrt(gx.square() + gy.square() + 1e-6)
    return magnitude / (magnitude.amax(dim=(2, 3), keepdim=True) + 1e-6)


class _Delta(nn.Module):
    def __init__(self, norm, base=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(5, 16, 3, padding=1, bias=False), norm(16), nn.ReLU(True),
                                 nn.Conv2d(16, base, 3, padding=1, bias=False), norm(base), nn.ReLU(True),
                                 nn.Conv2d(base, base, 3, padding=1, bias=False), norm(base), nn.ReLU(True),
                                 nn.Conv2d(base, 1, 1, bias=False))

    def forward(self, rgb, coarse):
        return self.net(torch.cat([rgb, coarse, _sobel(coarse)], 1))


class _AB(nn.Module):
    def __init__(self, norm, base=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(6, 16, 3, padding=1, bias=False), norm(16), nn.ReLU(True),
                                 nn.Conv2d(16, base, 3, padding=1, bias=False), norm(base), nn.ReLU(True),
                                 nn.Conv2d(base, base, 3, padding=1, bias=False), norm(base), nn.ReLU(True),
                                 nn.Conv2d(base, 2, 1))

    def forward(self, rgb, coarse, delta):
        ab = self.net(torch.cat([rgb, coarse, _sobel(coarse), delta], 1))
        return 1.0 + .2 * torch.tanh(ab[:, :1]), .2 * torch.tanh(ab[:, 1:])


class Refine3Stage(nn.Module):
    def __init__(self, variant="original"):
        super().__init__()
        norm = nn.BatchNorm2d if variant == "original" else lambda n: nn.GroupNorm(4 if n == 16 else 8, n)
        self.stage1, self.stage2, self.stage3 = _Delta(norm), _AB(norm), _AB(norm)
        self.variant = variant

    def forward(self, rgb, coarse):
        d1 = self.stage1(rgb, coarse); c1 = coarse + d1
        a2, b2 = self.stage2(rgb, c1, d1); d2 = a2 * d1 + b2
        c2 = coarse + d2
        a3, b3 = self.stage3(rgb, c2, d2); d3 = a3 * d2 + b3
        return {"delta1": d1, "delta2": d2, "delta3": d3, "final": coarse + d3}


def _exp6d_sobel(x):
    kx = x.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
    gx, gy = F.conv2d(x, kx, padding=1), F.conv2d(x, ky, padding=1)
    g = torch.sqrt(gx.square() + gy.square() + 1e-6)
    return g / (g.amax(dim=(2, 3), keepdim=True) + 1e-6)


class _Exp6DStage1(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1, bias=False), nn.GroupNorm(4, 16), nn.ReLU(True),
            nn.Conv2d(16, base, 3, padding=1, bias=False), nn.GroupNorm(8, base), nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1, bias=False), nn.GroupNorm(8, base), nn.ReLU(True),
            nn.Conv2d(base, 2, 1),
        )

    def forward(self, rgb, coarse):
        raw = self.net(torch.cat([rgb, coarse, _exp6d_sobel(coarse)], 1))
        gate = torch.sigmoid(raw[:, 1:2])
        return raw[:, :1] * gate


class _Exp6DStageAB(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1, bias=False), nn.GroupNorm(4, 16), nn.ReLU(True),
            nn.Conv2d(16, base, 3, padding=1, bias=False), nn.GroupNorm(8, base), nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1, bias=False), nn.GroupNorm(8, base), nn.ReLU(True),
            nn.Conv2d(base, 4, 1),
        )

    def forward(self, rgb, coarse, delta):
        raw = self.net(torch.cat([rgb, coarse, _exp6d_sobel(coarse), delta], 1))
        a = 1.0 + .2 * torch.tanh(raw[:, :1])
        b = .2 * torch.tanh(raw[:, 1:2])
        edge, smooth = torch.sigmoid(raw[:, 2:3]), torch.sigmoid(raw[:, 3:4])
        gate = edge * (1.0 - smooth) + .25 * smooth
        return a, b, gate


class Exp6DRefine3Stage(nn.Module):
    """Exp6D edge/smooth-gated RF, retrained for each frozen origin model."""
    def __init__(self, base=32):
        super().__init__()
        self.stage1, self.stage2, self.stage3 = _Exp6DStage1(base), _Exp6DStageAB(base), _Exp6DStageAB(base)

    def forward(self, rgb, coarse):
        d1 = self.stage1(rgb, coarse)
        c1 = coarse + d1
        a2, b2, g2 = self.stage2(rgb, c1, d1)
        proposal2 = a2 * d1 + b2
        d2 = d1 + g2 * (proposal2 - d1)
        c2 = coarse + d2
        a3, b3, g3 = self.stage3(rgb, c2, d2)
        proposal3 = a3 * d2 + b3
        d3 = d2 + g3 * (proposal3 - d2)
        return {"delta1": d1, "delta2": d2, "delta3": d3, "final": coarse + d3}


def exp6d_loss(outputs, target, coarse):
    """Compact edge/smooth/gradient loss used by the standalone Exp6D runs."""
    target_delta = target - coarse
    total = 0.0
    for key in ("delta1", "delta2", "delta3"):
        pred = outputs[key]
        edge = _exp6d_sobel(target)
        mask = (edge > edge.mean(dim=(2, 3), keepdim=True)).float().detach()
        smooth = 1.0 - mask
        pixel = (pred - target_delta).abs()
        grad = (_exp6d_sobel(pred) - _exp6d_sobel(target_delta)).abs()
        total = total + (0.5 * pixel * mask).mean() + (0.5 * pixel * smooth).mean() + (0.25 * grad * mask).mean()
    return total / 3.0


def build_refine(variant: str):
    if variant not in ("original", "exp4b", "exp6d"):
        raise ValueError(f"Unknown refine variant: {variant}")
    if variant == "exp6d":
        return Exp6DRefine3Stage()
    return Refine3Stage(variant)
