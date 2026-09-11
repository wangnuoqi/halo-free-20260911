from __future__ import annotations

import torch
import torch.nn.functional as F

from model import compute_sobel_grad


def batched_otsu_threshold(gradient: torch.Tensor, num_bins: int = 256, eps: float = 1e-6) -> torch.Tensor:
    flat = gradient.flatten(1)
    minimum = flat.min(dim=1).values
    maximum = flat.max(dim=1).values
    span = (maximum - minimum).clamp_min(eps)
    normalized = ((flat - minimum[:, None]) / span[:, None]).clamp(0.0, 1.0)
    bins = torch.clamp((normalized * (num_bins - 1)).long(), 0, num_bins - 1)
    histogram = torch.zeros(flat.shape[0], num_bins, device=gradient.device, dtype=gradient.dtype)
    histogram.scatter_add_(1, bins, torch.ones_like(flat))
    probability = histogram / histogram.sum(dim=1, keepdim=True).clamp_min(eps)
    values = torch.linspace(0.0, 1.0, num_bins, device=gradient.device, dtype=gradient.dtype)
    cumulative_probability = probability.cumsum(dim=1)
    cumulative_mean = (probability * values).cumsum(dim=1)
    global_mean = cumulative_mean[:, -1:]
    denominator = (cumulative_probability * (1.0 - cumulative_probability)).clamp_min(eps)
    between_class_variance = (global_mean * cumulative_probability - cumulative_mean).square() / denominator
    threshold_normalized = values[between_class_variance.argmax(dim=1)]
    return minimum + threshold_normalized * span


def masked_reduce(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1e-8)


def edge_guided_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gradient, _, _ = compute_sobel_grad(target, normalize=True)
    threshold = batched_otsu_threshold(gradient)
    edge_mask = (gradient > threshold.view(-1, 1, 1, 1)).float()
    no_edge = edge_mask.flatten(1).sum(dim=1) <= 0
    if no_edge.any():
        flat_mask = edge_mask.flatten(1)
        flat_mask[no_edge] = 0.0
        flat_mask[no_edge, gradient.flatten(1)[no_edge].argmax(dim=1)] = 1.0
        edge_mask = flat_mask.view_as(edge_mask)
    smooth_mask = 1.0 - edge_mask
    pixel_loss = (prediction - target).abs()
    edge_pixel = masked_reduce(pixel_loss, edge_mask)
    smooth_pixel = masked_reduce(pixel_loss, smooth_mask)
    _, pred_x, pred_y = compute_sobel_grad(prediction, normalize=True)
    _, target_x, target_y = compute_sobel_grad(target, normalize=True)
    edge_gradient = masked_reduce((pred_x - target_x).abs() + (pred_y - target_y).abs(), edge_mask)
    edge_ratio = (edge_gradient / (edge_pixel + 1e-6)).detach().clamp_min(1e-6)
    smooth_ratio = (smooth_pixel / (edge_pixel + 1e-6)).detach().clamp_min(1e-6)
    return edge_pixel + edge_gradient / edge_ratio + smooth_pixel / smooth_ratio


def cascade_loss(outputs: dict[str, torch.Tensor], noisy: torch.Tensor, clean: torch.Tensor):
    target_delta = clean - noisy
    stage_losses = (
        edge_guided_loss(outputs["delta1"], target_delta),
        edge_guided_loss(outputs["delta2"], target_delta),
        edge_guided_loss(outputs["delta3"], target_delta),
    )
    return 0.33 * sum(stage_losses), stage_losses


def staged_image_loss(outputs: dict[str, torch.Tensor], clean: torch.Tensor):
    stage_losses = (
        edge_guided_loss(outputs["stage1"], clean),
        edge_guided_loss(outputs["stage2"], clean),
        edge_guided_loss(outputs["final"], clean),
    )
    return 0.2 * stage_losses[0] + 0.3 * stage_losses[1] + 0.5 * stage_losses[2], stage_losses


def parametric_cascade_loss(outputs: dict[str, torch.Tensor], coarse: torch.Tensor, clean: torch.Tensor):
    target_delta = clean - coarse
    stage_losses = tuple(edge_guided_loss(outputs[key], target_delta) for key in ("delta1", "delta2", "delta3"))
    return 0.33 * sum(stage_losses), stage_losses
