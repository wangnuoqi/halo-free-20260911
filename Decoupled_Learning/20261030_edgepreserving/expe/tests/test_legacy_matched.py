from argparse import Namespace
import json
import sys

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from expe.train_legacy_matched import main, phase_learning_rates, phase_loss
from expe.matched_2x2_wintrain import command


@pytest.fixture
def args():
    return Namespace(lr_base=1e-4, lr_refine=1e-2,
                     lr_joint_coarse=3e-4, lr_joint_refine=3e-3)


def test_historical_learning_rates_are_assigned_to_the_correct_phase(args):
    assert phase_learning_rates("base", args) == [("coarse", 1e-4)]
    assert phase_learning_rates("refine", args) == [("refine", 1e-2)]
    assert phase_learning_rates("joint", args) == [
        ("coarse", 3e-4), ("refine", 3e-3)
    ]


def test_refine_loss_excludes_joint_coarse_penalty_and_joint_adds_it():
    coarse = torch.rand(2, 1, 12, 12, requires_grad=True) - 0.5
    outputs = {"coarse0": coarse}
    target = torch.rand(2, 1, 12, 12)
    for index in range(1, 4):
        outputs[f"delta{index}"] = torch.rand_like(coarse, requires_grad=True) * 0.05
    refine = phase_loss(outputs, target, "refine")
    joint = phase_loss(outputs, target, "joint")
    coarse_penalty = 0.5 * F.smooth_l1_loss(coarse, target - 0.5)
    torch.testing.assert_close(joint - refine, coarse_penalty)


def test_matched_budget_counts_optimizer_steps_and_exposures(tmp_path, monkeypatch):
    folders = ["traininput-l0p500-a1p400_downsample",
               "traininput-l0p500-a1p400", "trainlabel-l0p500-a1p400"]
    for folder in folders:
        (tmp_path / folder).mkdir()
    for index in range(3):
        stem = f"{index:04d}"
        lr = np.full((8, 8), 0.4 + index * 0.01, dtype=np.float32)
        hr = np.full((16, 16), 0.4 + index * 0.01, dtype=np.float32)
        gt = np.full((16, 16), 0.5 + index * 0.01, dtype=np.float32)
        cv2.imwrite(str(tmp_path / folders[0] / f"{stem}-log.tiff"), lr)
        cv2.imwrite(str(tmp_path / folders[1] / f"{stem}-log.tiff"), hr)
        cv2.imwrite(str(tmp_path / folders[2] / f"{stem}-WLS-l0p500-a1p400.tiff"), gt)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train_legacy_matched.py", "--variant", "legacy_matched_e0",
        "--phase", "base", "--parallelism", "single", "--device", "cpu",
        "--data-root", str(tmp_path), "--output-dir", str(output),
        "--crop-size", "8", "--batch-size-per-rank", "1", "--grad-accumulation", "2",
        "--total-iter", "2", "--val-every", "1", "--save-every", "1",
        "--val-count", "1", "--num-workers", "0"])
    main()
    state = torch.load(output / "final.pth", map_location="cpu", weights_only=False)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert state["iteration"] == 2
    assert protocol["global_batch_size"] == 2
    assert protocol["sample_exposures"] == 4
    assert protocol["scheduler_milestones"] == [1, 2]
    assert (output / "best.pth").is_file()


@pytest.mark.parametrize("cell", ["new_e0", "old_e0", "old_e1"])
def test_windows_commands_match_e1_base_exposure(cell):
    result = command(cell)
    assert result.startswith("wintrain '")
    assert '--parallelism ddp' in result
    assert '--nproc_per_node=2' in result
    assert 'TORCH_DDP_BACKEND="gloo"' in result
    assert '--grad-accumulation 2' in result
    assert '--total-iter 60000' in result
    assert '--batch-size-per-gpu 16' in result or '--batch-size-per-rank 16' in result
    assert 'legacy_matched_exposure3840000_v5' in result or 'seed42_exposure3840000' in result
