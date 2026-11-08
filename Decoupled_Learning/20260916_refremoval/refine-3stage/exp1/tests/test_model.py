from pathlib import Path
import sys

import torch


EXP1_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP1_ROOT))

from model import CoarseEdgeGatedRefine3Stage


def test_three_stage_returns_progressive_outputs_and_uses_coarse_edges_only():
    torch.manual_seed(7)
    model = CoarseEdgeGatedRefine3Stage(base_channels=8, num_blocks=1)
    coarse = torch.rand(2, 3, 16, 16)
    blended = torch.rand(2, 3, 16, 16)
    prior = torch.rand(2, 1, 16, 16)

    outputs = model(coarse, blended, prior)

    assert set(outputs) >= {
        "final", "coarse1", "coarse2", "delta1", "delta2", "delta3",
        "gate1", "gate2", "gate3", "edge0", "edge1", "edge2",
        "detail1", "detail2", "detail3",
    }
    assert outputs["final"].shape == coarse.shape
    assert outputs["gate1"].shape == prior.shape
    assert outputs["edge0"].shape == prior.shape
    assert torch.all((outputs["gate1"] >= 1) & (outputs["gate1"] <= 1.25))
    assert torch.allclose(outputs["final"], coarse + outputs["delta3"], atol=1e-6)


def test_zero_gate_removes_blended_high_frequency_from_stage_predictor_input():
    model = CoarseEdgeGatedRefine3Stage(base_channels=8, num_blocks=1)
    coarse = torch.zeros(1, 3, 16, 16)
    blended = torch.zeros_like(coarse)
    blended[:, :, ::2, 1::2] = 1.0
    prior = torch.zeros(1, 1, 16, 16)

    detail = model.gated_missing_detail(coarse, blended, torch.zeros_like(prior))

    assert torch.count_nonzero(detail) == 0
