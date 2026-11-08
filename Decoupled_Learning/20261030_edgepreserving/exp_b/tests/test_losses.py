import torch

from exp_b.src.train_exp_b import (
    base_loss,
    destroy_process_group_safely,
    edge_guided_loss,
    legacy_cascade_loss,
)


def test_edge_guided_loss_is_finite_on_constant_images():
    target = torch.full((2, 1, 16, 16), 0.25)
    pred = target.clone().requires_grad_(True)
    loss = edge_guided_loss(pred, target)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_legacy_loss_uses_three_delta_outputs_and_coarse_fidelity():
    target = torch.rand(2, 1, 16, 16)
    coarse = torch.rand_like(target)
    outputs = {
        "coarse0": coarse,
        "delta1": torch.zeros_like(target),
        "delta2": torch.zeros_like(target),
        "delta3": torch.zeros_like(target),
    }
    loss, parts = legacy_cascade_loss(outputs, target)
    expected = 0.1 * parts[0] + 0.3 * parts[1] + 0.6 * parts[2]
    expected = expected + 0.5 * torch.nn.functional.smooth_l1_loss(coarse, target)
    torch.testing.assert_close(loss, expected)


def test_base_loss_includes_pixel_and_sobel_terms():
    pred = torch.rand(1, 1, 16, 16, requires_grad=True)
    target = torch.rand_like(pred)
    loss = base_loss(pred, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_process_group_cleanup_failure_is_reported_without_masking_completed_work(
    monkeypatch, capsys
):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def fail_cleanup():
        raise RuntimeError("simulated cleanup failure")

    monkeypatch.setattr(dist, "destroy_process_group", fail_cleanup)
    destroy_process_group_safely()
    assert "simulated cleanup failure" in capsys.readouterr().out
