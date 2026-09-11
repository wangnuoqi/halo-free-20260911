from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sobel_grad(x: torch.Tensor, eps: float = 1e-6, normalize: bool = False):
    sobel_x = x.new_tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]])
    sobel_y = x.new_tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]])
    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)
    grad = torch.sqrt(grad_x.square() + grad_y.square() + eps)
    if normalize:
        grad = grad / (grad.amax(dim=(2, 3), keepdim=True) + eps)
    return grad, grad_x, grad_y


class Stage1DeltaNet(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, base_channels, 3, padding=1, bias=False), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1, bias=False),
        )

    def forward(self, noisy: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        grad, _, _ = compute_sobel_grad(estimate, normalize=True)
        return self.net(torch.cat([noisy, estimate, grad], dim=1))


class ABAdjustNet(nn.Module):
    def __init__(self, base_channels: int = 32, a_scale: float = 0.2, b_scale: float = 0.2) -> None:
        super().__init__()
        self.a_scale = a_scale
        self.b_scale = b_scale
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, base_channels, 3, padding=1, bias=False), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 2, 1, bias=True),
        )

    def forward(self, noisy: torch.Tensor, estimate: torch.Tensor, previous_delta: torch.Tensor):
        grad, _, _ = compute_sobel_grad(estimate, normalize=True)
        a_raw, b_raw = torch.chunk(self.net(torch.cat([noisy, estimate, grad, previous_delta], dim=1)), 2, dim=1)
        return 1.0 + self.a_scale * torch.tanh(a_raw), self.b_scale * torch.tanh(b_raw)


class IndependentResidual3Stage(nn.Module):
    """Three fresh residual predictions, conditioned on the frozen coarse result."""

    def __init__(self, base_channels: int = 32, a_scale: float = 0.2, b_scale: float = 0.2) -> None:
        super().__init__()
        self.stage1 = Stage1DeltaNet(base_channels)
        self.stage2 = Stage1DeltaNet(base_channels)
        self.stage3 = Stage1DeltaNet(base_channels)

    def forward(self, noisy: torch.Tensor, coarse: torch.Tensor):
        delta1 = self.stage1(noisy, coarse)
        estimate1 = coarse + delta1
        delta2 = self.stage2(noisy, estimate1)
        estimate2 = estimate1 + delta2
        delta3 = self.stage3(noisy, estimate2)
        final = estimate2 + delta3
        return {
            "delta1": delta1, "stage1": estimate1,
            "delta2": delta2, "stage2": estimate2,
            "delta3": delta3, "final": final,
        }
