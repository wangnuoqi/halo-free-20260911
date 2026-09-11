import torch
from torch.nn import functional as F

from exp2.A.convguided_model import SurveyDGF


def test_exp2_a_forward_shape_and_smoothing_parameter():
    model = SurveyDGF()
    depth_lr = torch.randn(2, 1, 8, 10)
    rgb_hr = torch.randn(2, 3, 32, 40)

    output = model(depth_lr, rgb_hr)

    assert output.shape == (2, 1, 32, 40)
    assert hasattr(model, "guide_smoothing_logit")
    assert model.guide_smoothing_logit.ndim == 0


def test_exp2_a_initial_alpha_and_reflect_blurred_guide_mix():
    model = SurveyDGF()
    raw_guide = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

    guide = model._build_guide(raw_guide)

    alpha = torch.sigmoid(torch.tensor(-4.0))
    expected_blur = F.avg_pool2d(F.pad(raw_guide, (1, 1, 1, 1), mode="reflect"), 3, 1)
    assert torch.allclose(torch.sigmoid(model.guide_smoothing_logit), alpha)
    assert torch.allclose(guide, (1 - alpha) * raw_guide + alpha * expected_blur)
