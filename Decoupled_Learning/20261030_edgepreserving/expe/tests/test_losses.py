import torch

from expe.losses import experiment_loss


def test_base_loss_matches_legacy_coarse_balancing():
    target = torch.zeros(1, 1, 8, 8)
    prediction = torch.linspace(0, 0.4, 8).view(1, 1, 1, 8).expand_as(target).clone()
    prediction.requires_grad_()
    total, pixel, gradient = experiment_loss(prediction, target, {}, "base")
    assert gradient > 0
    torch.testing.assert_close(total.detach(), 2 * pixel, rtol=1e-4, atol=1e-7)
    total.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_base_loss_has_no_edge_penalty_for_constant_offset():
    target = torch.zeros(1, 1, 8, 8)
    prediction = torch.full_like(target, 0.2)
    total, pixel, gradient = experiment_loss(prediction, target, {}, "base")
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-7, rtol=0)
    torch.testing.assert_close(total, pixel)
