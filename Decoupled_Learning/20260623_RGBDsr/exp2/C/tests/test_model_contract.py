import torch
from torch import nn

from exp2.C.convguided_model import SurveyDGF


def test_exp2_c_returns_hr_depth_and_exposes_detail_gate():
    model = SurveyDGF()
    depth_lr = torch.rand(2, 1, 8, 10)
    rgb_hr = torch.rand(2, 3, 32, 40)

    output = model(depth_lr, rgb_hr)

    assert output.shape == (2, 1, 32, 40)
    assert model.last_detail_gate.shape == (2, 1, 32, 40)
    convs = [module for module in model.detail_gate if isinstance(module, nn.Conv2d)]
    assert [(conv.in_channels, conv.out_channels) for conv in convs] == [(4, 16), (16, 1)]


def test_exp2_c_gate_controls_only_rgb_high_frequency_detail():
    model = SurveyDGF()
    assert torch.allclose(model.detail_gate[2].bias, torch.tensor([4.0]))
    rgb_gray = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
    depth_up = torch.full_like(rgb_gray, 9.0)

    with torch.no_grad():
        for layer in model.detail_gate:
            if isinstance(layer, nn.Conv2d):
                layer.weight.zero_()
                layer.bias.zero_()

    guide, gate = model._build_guide(rgb_gray, depth_up)
    expected_blur = torch.tensor([[[[2.0, 5.0 / 3.0], [4.0 / 3.0, 1.0]]]])

    assert torch.allclose(gate, torch.full_like(gate, 0.5))
    assert torch.allclose(guide, expected_blur + 0.5 * (rgb_gray - expected_blur))
    assert not torch.any(torch.isclose(guide, depth_up))
