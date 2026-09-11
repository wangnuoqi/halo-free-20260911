from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_source = Path(__file__).resolve().parent.parent / "exp4" / "loss.py"
sys.path.insert(0, str(_source.parent))
_spec = importlib.util.spec_from_file_location("exp4_loss", _source)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load baseline loss: {_source}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def weighted_parametric_cascade_loss(
    outputs: dict[str, torch.Tensor],
    coarse: torch.Tensor,
    clean: torch.Tensor,
    *,
    stage_weights: tuple[float, float, float] = (0.1, 0.2, 0.7),
    final_mse_weight: float = 0.0,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    if len(stage_weights) != 3 or min(stage_weights) < 0:
        raise ValueError("stage_weights must contain three non-negative values")
    target_delta = clean - coarse
    stage_losses = tuple(_module.edge_guided_loss(outputs[key], target_delta) for key in ("delta1", "delta2", "delta3"))
    weighted_loss = sum(weight * loss for weight, loss in zip(stage_weights, stage_losses))
    return weighted_loss + final_mse_weight * F.mse_loss(outputs["final"], clean), stage_losses
