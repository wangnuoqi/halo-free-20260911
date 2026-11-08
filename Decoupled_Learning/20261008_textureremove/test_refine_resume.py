from pathlib import Path

import torch

from train_refine_rtv import restore_refine_checkpoint


def make_training_state():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
    return model, optimizer, scheduler


def test_resume_from_legacy_best_weights(tmp_path: Path):
    source, _, _ = make_training_state()
    with torch.no_grad():
        source.weight.fill_(0.25)
    best_path = tmp_path / "refine_rtv_best.pth"
    torch.save({f"module.{key}": value for key, value in source.state_dict().items()}, best_path)

    model, optimizer, scheduler = make_training_state()
    start_epoch, best, resumed_from_best = restore_refine_checkpoint(
        model, optimizer, scheduler, tmp_path / "refine_rtv_full.pth", best_path, torch.device("cpu")
    )

    assert start_epoch == 0
    assert best == float("inf")
    assert resumed_from_best
    assert torch.equal(model.weight, source.weight)


def test_resume_from_full_checkpoint_prefers_exact_state(tmp_path: Path):
    source, source_optimizer, source_scheduler = make_training_state()
    with torch.no_grad():
        source.weight.fill_(0.75)
    source_optimizer.param_groups[0]["lr"] = 0.0005
    source_scheduler.last_epoch = 34
    full_path = tmp_path / "refine_rtv_full.pth"
    torch.save({
        "epoch": 35,
        "model_state_dict": {f"module.{key}": value for key, value in source.state_dict().items()},
        "optimizer_state_dict": source_optimizer.state_dict(),
        "scheduler_state_dict": source_scheduler.state_dict(),
        "best_loss": 0.002,
    }, full_path)

    model, optimizer, scheduler = make_training_state()
    start_epoch, best, resumed_from_best = restore_refine_checkpoint(
        model, optimizer, scheduler, full_path, tmp_path / "missing_best.pth", torch.device("cpu")
    )

    assert start_epoch == 35
    assert best == 0.002
    assert not resumed_from_best
    assert torch.equal(model.weight, source.weight)
    assert optimizer.param_groups[0]["lr"] == 0.0005
    assert scheduler.last_epoch == 34
