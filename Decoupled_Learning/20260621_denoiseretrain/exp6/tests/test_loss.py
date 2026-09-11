import torch

from loss import weighted_parametric_cascade_loss


def test_final_mse_weight_is_applied_only_to_final_image():
    clean = torch.ones(1, 1, 4, 4)
    coarse = torch.zeros_like(clean)
    outputs = {key: torch.zeros_like(clean) for key in ("delta1", "delta2", "delta3", "final")}
    base, _ = weighted_parametric_cascade_loss(outputs, coarse, clean, final_mse_weight=0.0)
    with_mse, _ = weighted_parametric_cascade_loss(outputs, coarse, clean, final_mse_weight=5.0)
    assert torch.isclose(with_mse - base, torch.tensor(5.0))


def test_later_stage_weighting_changes_total_loss():
    coarse = torch.zeros(1, 1, 4, 4)
    clean = torch.ones_like(coarse)
    outputs = {
        "delta1": torch.zeros_like(clean),
        "delta2": torch.full_like(clean, 0.5),
        "delta3": torch.ones_like(clean),
        "final": clean,
    }
    early, _ = weighted_parametric_cascade_loss(outputs, coarse, clean, stage_weights=(0.7, 0.2, 0.1))
    late, _ = weighted_parametric_cascade_loss(outputs, coarse, clean, stage_weights=(0.1, 0.2, 0.7))
    assert late < early
