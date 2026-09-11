import torch
from torch import nn
from torch.nn import functional as F

from exp2.B.convguided_model import SurveyDGF


def test_exp2_b_forward_shape_confidence_and_head():
    model = SurveyDGF()
    depth_lr = torch.randn(2, 1, 8, 10)
    rgb_hr = torch.randn(2, 3, 32, 40)

    output = model(depth_lr, rgb_hr)

    assert output.shape == (2, 1, 32, 40)
    assert model.last_confidence.shape == (2, 1, 32, 40)
    convs = [module for module in model.confidence_head if isinstance(module, nn.Conv2d)]
    assert [conv.in_channels for conv in convs] == [4, 16]
    assert [conv.out_channels for conv in convs] == [16, 1]


def test_exp2_b_confidence_bias_bounds_and_sobel_driven_guide_blend():
    model = SurveyDGF()
    assert torch.allclose(model.confidence_head[2].bias, torch.tensor([2.0]))
    with torch.no_grad():
        model.confidence_head[0].weight.zero_()
        model.confidence_head[0].bias.zero_()
        model.confidence_head[0].weight[0, 2, 1, 1] = 1.0
        model.confidence_head[2].weight.zero_()
        model.confidence_head[2].bias.zero_()
        model.confidence_head[2].weight[0, 0, 1, 1] = 1.0
    rgb_gray = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]])
    depth_up = torch.full_like(rgb_gray, 2.0)

    guide, confidence = model._build_guide(rgb_gray, depth_up)

    sobel_rgb = model._gradient_magnitude(rgb_gray)
    assert torch.allclose(confidence, torch.sigmoid(sobel_rgb))
    assert torch.all((confidence >= 0) & (confidence <= 1))
    assert torch.allclose(guide, confidence * rgb_gray + (1 - confidence) * depth_up)
