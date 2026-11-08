"""Three-stage coarse-edge-preserving FUMO refinement."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _lowpass(image: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)


def _highpass(image: torch.Tensor) -> torch.Tensor:
    return image - _lowpass(image)


def _coarse_edge(image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    gray = image.mean(dim=1, keepdim=True)
    kx = gray.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
    ky = gray.new_tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
    gx, gy = F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)
    edge = torch.sqrt(gx.square() + gy.square() + eps)
    return edge / edge.amax(dim=(2, 3), keepdim=True).clamp_min(eps)


def _map01(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize a spatial prior independently per image to [0, 1]."""
    lo = value.amin(dim=(2, 3), keepdim=True)
    hi = value.amax(dim=(2, 3), keepdim=True)
    return ((value - lo) / (hi - lo).clamp_min(eps)).clamp(0., 1.)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        expanded = channels * 2
        self.norm1, self.norm2 = nn.GroupNorm(1, channels), nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv2d(channels, expanded, 1)
        self.dw = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded)
        self.pw2 = nn.Conv2d(channels, channels, 1)
        self.ffn1, self.ffn2 = nn.Conv2d(channels, expanded, 1), nn.Conv2d(channels, channels, 1)
        self.gate1, self.gate2 = SimpleGate(), SimpleGate()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.beta * self.pw2(self.gate1(self.dw(self.pw1(self.norm1(x)))))
        return x + self.gamma * self.ffn2(self.gate2(self.ffn1(self.norm2(x))))


class ConvStack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int, blocks: int) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, width, 3, padding=1)
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(blocks)])
        self.tail = nn.Conv2d(width, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        return self.tail(x + self.body(x))


class GatedRefineStage(nn.Module):
    """Three-stage residual block with explicit prior/detail gate."""
    def __init__(self, width: int, blocks: int, residual_scale: float, gate_beta: float) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.gate_beta = gate_beta
        # current RGB(3), coarse edge(1), prior(1), gated detail(3), LF gap(3), delta_prev(3)
        self.predictor = ConvStack(14, 3, width, blocks)

    def forward(self, current: torch.Tensor, blended: torch.Tensor, prior: torch.Tensor, delta_prev: torch.Tensor):
        edge = _coarse_edge(current)
        missing_hf = _highpass(blended) - _highpass(current)
        low_frequency_gap = _lowpass(blended) - _lowpass(current)
        # Paper-style spatial enhancement: g = 1 + beta * P_int * P_hf.
        # P_int is the reflection prior; P_hf measures local input detail richness.
        p_int = _map01(prior.mean(1, keepdim=True))
        p_hf = _map01(_highpass(blended).abs().mean(1, keepdim=True))
        gate = 1.0 + self.gate_beta * p_int * p_hf
        detail = gate * missing_hf
        predictor_input = torch.cat([current, edge, prior, detail, low_frequency_gap, delta_prev], 1)
        residual = self.residual_scale * torch.tanh(self.predictor(predictor_input))
        return residual, gate, edge, detail


class CoarseEdgeGatedRefine3Stage(nn.Module):
    """Three residual stages; only current/coarse edges define structure."""
    def __init__(self, base_channels: int = 48, num_blocks: int = 4, residual_scale: float = 0.20, gate_beta: float = 0.25) -> None:
        super().__init__()
        self.stage1 = GatedRefineStage(base_channels, num_blocks, residual_scale, gate_beta)
        self.stage2 = GatedRefineStage(base_channels, num_blocks, residual_scale, gate_beta)
        self.stage3 = GatedRefineStage(base_channels, num_blocks, residual_scale, gate_beta)

    @staticmethod
    def gated_missing_detail(current: torch.Tensor, blended: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return gate * (_highpass(blended) - _highpass(current))

    def forward(self, coarse: torch.Tensor, blended: torch.Tensor, prior: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = torch.zeros_like(coarse)
        r1, g1, e0, d1 = self.stage1(coarse, blended, prior, zero)
        delta1, coarse1 = r1, coarse + r1
        r2, g2, e1, d2 = self.stage2(coarse1, blended, prior, delta1)
        delta2, coarse2 = delta1 + r2, coarse + delta1 + r2
        r3, g3, e2, d3 = self.stage3(coarse2, blended, prior, delta2)
        delta3 = delta2 + r3
        return {
            "final": (coarse + delta3).clamp(0., 1.), "coarse1": coarse1.clamp(0., 1.), "coarse2": coarse2.clamp(0., 1.),
            "delta1": delta1, "delta2": delta2, "delta3": delta3,
            "residual1": r1, "residual2": r2, "residual3": r3,
            "gate1": g1, "gate2": g2, "gate3": g3,
            "edge0": e0, "edge1": e1, "edge2": e2, "detail1": d1, "detail2": d2, "detail3": d3,
        }
