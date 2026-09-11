from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sobel_grad(
    x: torch.Tensor, eps: float = 1e-6, normalize: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sobel_x = x.new_tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]])
    sobel_y = x.new_tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]])
    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)
    grad = torch.sqrt(grad_x.square() + grad_y.square() + eps)
    if normalize:
        grad = grad / (grad.amax(dim=(2, 3), keepdim=True) + eps)
    return grad, grad_x, grad_y


def deep_stack(in_channels: int, out_channels: int, base_channels: int = 32) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        nn.Conv2d(16, base_channels, 3, padding=1, bias=False), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
        nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
        nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
        nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
        nn.Conv2d(base_channels, out_channels, 1, bias=out_channels == 2),
    )


class DeepDeltaNet(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.net = deep_stack(3, 1, base_channels)

    def forward(self, noisy: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        gradient, _, _ = compute_sobel_grad(estimate, normalize=True)
        return self.net(torch.cat([noisy, estimate, gradient], dim=1))


class DeepABAdjustNet(nn.Module):
    def __init__(self, base_channels: int = 32, a_scale: float = 0.2, b_scale: float = 0.2) -> None:
        super().__init__()
        self.a_scale = a_scale
        self.b_scale = b_scale
        self.net = deep_stack(4, 2, base_channels)

    def forward(self, noisy: torch.Tensor, estimate: torch.Tensor, previous_delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gradient, _, _ = compute_sobel_grad(estimate, normalize=True)
        a_raw, b_raw = torch.chunk(
            self.net(torch.cat([noisy, estimate, gradient, previous_delta], dim=1)),
            2,
            dim=1,
        )
        return 1.0 + self.a_scale * torch.tanh(a_raw), self.b_scale * torch.tanh(b_raw)


class IdentityInitializedDeepParametricCascade3Stage(nn.Module):
    """Experiment 5A: five-layer parameterized cascade initialized as identity."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.stage1 = DeepDeltaNet(base_channels)
        self.stage2 = DeepABAdjustNet(base_channels)
        self.stage3 = DeepABAdjustNet(base_channels)
        self._initialize_identity()

    def _initialize_identity(self) -> None:
        for stage in (self.stage1, self.stage2, self.stage3):
            output_layer = stage.net[-1]
            nn.init.zeros_(output_layer.weight)
            if output_layer.bias is not None:
                nn.init.zeros_(output_layer.bias)

    def forward(self, noisy: torch.Tensor, coarse: torch.Tensor) -> dict[str, torch.Tensor]:
        delta1 = self.stage1(noisy, coarse)
        estimate1 = coarse + delta1
        a2, b2 = self.stage2(noisy, estimate1, delta1)
        delta2 = a2 * delta1 + b2
        estimate2 = coarse + delta2
        a3, b3 = self.stage3(noisy, estimate2, delta2)
        delta3 = a3 * delta2 + b3
        return {"delta1": delta1, "stage1": estimate1, "delta2": delta2, "stage2": estimate2, "delta3": delta3, "final": coarse + delta3}
