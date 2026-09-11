from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from exp7_8_common import BSDSOnline, OfficialPatchDataset, coarse_loss, joint_loss


def _image(path, height, width):
    cv2.imwrite(str(path), np.full((height, width), 127, dtype=np.uint8))


def test_dncnn_dense_grid_keeps_all_four_scales(tmp_path):
    _image(tmp_path / "one.png", 50, 50)
    dataset = OfficialPatchDataset(tmp_path, "dncnn", seed=3)

    # 40x40 patch with stride 10: scale 1.0 has four coordinates;
    # 0.9 and 0.8 each have one coordinate; 0.7 is too small.
    assert len(dataset) == 6
    sample = dataset[0]
    assert sample["clean"].shape == (1, 40, 40)


def test_dense_grid_caches_resized_source_images(tmp_path):
    _image(tmp_path / "one.png", 50, 50)
    dataset = OfficialPatchDataset(tmp_path, "dncnn", seed=3)

    dataset[0]
    dataset[1]

    assert len(dataset._resized_cache) == 1


def test_patch_dataset_groups_repeated_patches_by_source_image(tmp_path):
    _image(tmp_path / "one.png", 50, 50)
    _image(tmp_path / "two.png", 50, 50)
    dataset = BSDSOnline(tmp_path, "patch", patch_size=40, patches_per_image=2, train=True)

    assert dataset[0]["name"] == dataset[1]["name"]
    assert dataset[2]["name"] == dataset[3]["name"]
    assert dataset[0]["name"] != dataset[2]["name"]


def test_ffdnet_dense_grid_rebuilds_scale_deterministically(tmp_path):
    _image(tmp_path / "one.png", 140, 140)
    dataset = OfficialPatchDataset(tmp_path, "ffdnet", seed=7)
    first_epoch = dataset.epoch
    dataset.rebuild(epoch=10)

    assert len(dataset) > 0
    assert dataset.epoch == first_epoch + 10
    assert dataset[0]["clean"].shape == (1, 70, 70)


def test_joint_loss_contains_coarse_fidelity_term():
    rf_loss = torch.tensor(2.0)
    coarse = torch.tensor([[[[0.0]]]])
    clean = torch.tensor([[[[1.0]]]])
    mask = torch.ones_like(clean)

    total, fidelity = joint_loss(rf_loss, coarse, clean, mask)

    assert torch.isclose(fidelity, torch.tensor(0.5))
    assert torch.isclose(total, torch.tensor(2.25))


def test_coarse_loss_ablation_selects_mse_and_smooth_terms():
    coarse = torch.tensor([[[[0.0]]]])
    clean = torch.tensor([[[[1.0]]]])
    mask = torch.ones_like(clean)

    assert torch.isclose(coarse_loss(coarse, clean, mask, 'mse'), torch.tensor(1.0))
    assert torch.isclose(coarse_loss(coarse, clean, mask, 'mse_smooth'), torch.tensor(1.5))
