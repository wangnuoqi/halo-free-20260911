import pytest
import torch

from exp_b.src.model_exp_b import (
    VIVResidualBlock,
    VIVCompatibleLR,
    FullExpB,
    validate_phase,
)
from exp_b.src.train_exp_b import forward_phase


def test_viv_lr_keeps_single_channel_raw_full_resolution_contract():
    model = VIVCompatibleLR()
    x = torch.rand(2, 1, 64, 64) - 0.5
    y = model(x)
    assert y.shape == x.shape
    # Raw LR predictions must not be sigmoid-bounded by the model interface.
    assert model.output_mode == "raw"


def test_zero_initialized_viv_residual_starts_as_original_identity_block():
    block = VIVResidualBlock(channels=24, dilation=4)
    x = torch.randn(2, 24, 16, 16)
    expected = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
    torch.testing.assert_close(block(x), expected)


def test_full_exp_b_returns_all_stage_images_and_deltas_at_hr_resolution():
    model = FullExpB().eval()
    lr = torch.rand(1, 1, 32, 32) - 0.5
    hr = torch.rand(1, 1, 64, 64) - 0.5
    with torch.inference_mode():
        result = model(lr, hr)
    for key in ("coarse0", "coarse1", "coarse2", "final", "delta1", "delta2", "delta3"):
        assert result[key].shape == hr.shape
        assert torch.isfinite(result[key]).all()


def test_phase_names_are_validated():
    assert validate_phase("base") == "base"
    assert validate_phase("refine") == "refine"
    assert validate_phase("joint") == "joint"
    with pytest.raises(ValueError, match="phase"):
        validate_phase("anything")


def test_refine_phase_returns_unit_range_images_and_backpropagates_only_refine():
    model = FullExpB().eval()
    lr = torch.rand(1, 1, 8, 8) - 0.5
    hr = torch.rand(1, 1, 16, 16) - 0.5
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.refine.parameters():
        parameter.requires_grad_(True)
    with torch.enable_grad():
        outputs = forward_phase("refine", model, model.refine, lr + 0.5, hr + 0.5)
        assert outputs["final"].min() >= 0.0
        assert outputs["final"].max() <= 1.0
        outputs["final"].mean().backward()
    assert any(parameter.grad is not None for parameter in model.refine.parameters())
    assert all(parameter.grad is None for parameter in model.base.parameters())
