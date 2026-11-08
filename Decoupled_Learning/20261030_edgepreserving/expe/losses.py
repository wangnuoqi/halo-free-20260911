"""Comparable base, edge, and staged losses for Exp E affine ablations."""

import torch
from torch.nn import functional as F

from expe.model import otsu_edge_map


LOSS_MODES = ("base", "edge", "staged", "staged_regularized")
ROUTE_SUPERVISION_WEIGHT = 1e-4
ROUTE_SUPERVISION_TEMPERATURE = 0.05


def route_supervision_loss(auxiliary: dict, guide: torch.Tensor,
                           target: torch.Tensor) -> torch.Tensor:
    """Teach the gate to predict which affine candidate is closer to GT.

    The GT-derived routing target is detached.  Consequently this auxiliary
    term trains the gate, while the ordinary reconstruction loss remains the
    only term that trains the candidate branches from their final output.
    ``guide`` and ``target`` are HR tensors; candidates and gate weights live
    on the LR grid.
    """
    candidate_a = auxiliary["candidate_a"].float()
    candidate_b = auxiliary["candidate_b"].float()
    weights = auxiliary["box_weights"].float()
    if candidate_a.ndim != 4 or candidate_a.shape != candidate_b.shape:
        raise ValueError("candidate affine tensors must have shape [B,2,H,W]")
    if weights.shape != candidate_a.shape:
        raise ValueError("route supervision requires two gate weights aligned to candidates")
    if candidate_a.shape[1] != 2:
        raise ValueError("route supervision currently supports exactly two affine candidates")
    if guide.shape != target.shape:
        raise ValueError("guide and target must have the same HR shape")

    x_hr = guide.float() - 0.5
    candidates = []
    for index in range(2):
        a_hr = F.interpolate(candidate_a[:, index:index + 1], guide.shape[-2:],
                             mode="bilinear", align_corners=True)
        b_hr = F.interpolate(candidate_b[:, index:index + 1], guide.shape[-2:],
                             mode="bilinear", align_corners=True)
        candidates.append((a_hr * x_hr + b_hr).clamp(0.0, 1.0))
    candidates = torch.cat(candidates, dim=1)
    errors_hr = (candidates - target.float()).abs()
    errors_lr = F.interpolate(
        errors_hr, size=candidate_a.shape[-2:], mode="area"
    )
    target_weights = torch.softmax(
        -errors_lr / ROUTE_SUPERVISION_TEMPERATURE, dim=1
    ).detach()
    predicted_weights = weights.clamp_min(1e-6)
    return -(target_weights * predicted_weights.log()).sum(dim=1).mean()


def _legacy_coarse_loss(prediction, target):
    prediction, target = prediction.float(), target.float()
    pixel = F.mse_loss(prediction, target)
    kernel_x = prediction.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-2, -1)
    pred = F.pad(prediction, (1, 1, 1, 1), mode="reflect")
    truth = F.pad(target, (1, 1, 1, 1), mode="reflect")
    gradient = ((F.conv2d(pred, kernel_x) - F.conv2d(truth, kernel_x)).abs()
                + (F.conv2d(pred, kernel_y) - F.conv2d(truth, kernel_y)).abs()).mean()
    eps = 1e-8
    scale = (gradient / (pixel + eps)).detach().clamp_min(0.1)
    return pixel + gradient / scale, pixel, gradient


def _image_loss(prediction, target, edge_mask):
    error = torch.sqrt((prediction.float() - target.float()).square() + 1e-6)
    if edge_mask is None:
        pixel = error.mean()
    else:
        weights = 1.0 + 2.0 * edge_mask
        pixel = (weights * error).sum() / weights.sum()
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    true_dx = target[..., :, 1:] - target[..., :, :-1]
    true_dy = target[..., 1:, :] - target[..., :-1, :]
    dx, dy = (pred_dx - true_dx).abs(), (pred_dy - true_dy).abs()
    if edge_mask is None:
        gradient = dx.mean() + dy.mean()
    else:
        wx = 1.0 + 2.0 * edge_mask[..., :, 1:]
        wy = 1.0 + 2.0 * edge_mask[..., 1:, :]
        gradient = (wx * dx).sum() / wx.sum() + (wy * dy).sum() / wy.sum()
    return pixel + 0.05 * gradient, pixel, gradient


def experiment_loss(prediction: torch.Tensor, target: torch.Tensor,
                    auxiliary: dict, mode: str):
    if mode not in LOSS_MODES:
        raise ValueError(f"Unknown loss mode: {mode}")
    target = target.float()
    if mode == "base":
        total, pixel, gradient = _legacy_coarse_loss(prediction, target)
        return total, pixel.detach(), gradient.detach()
    edge_mask = otsu_edge_map(target)
    if mode in {"staged", "staged_regularized"}:
        stages = auxiliary["stages"]
        weights = torch.arange(1, len(stages) + 1, device=target.device, dtype=target.dtype)
        weights = weights / weights.sum()
        components = [_image_loss(stage.float(), target, edge_mask) for stage in stages]
        total = sum(weight * component[0] for weight, component in zip(weights, components))
        pixel = sum(weight * component[1] for weight, component in zip(weights, components))
        gradient = sum(weight * component[2] for weight, component in zip(weights, components))
    else:
        total, pixel, gradient = _image_loss(prediction.float(), target, edge_mask)
    if mode == "staged_regularized":
        changes = auxiliary.get("delta_a", []) + auxiliary.get("delta_b", [])
        if changes:
            total = total + 1e-3 * sum(change.abs().mean() for change in changes) / len(changes)
    return total, pixel.detach(), gradient.detach()
