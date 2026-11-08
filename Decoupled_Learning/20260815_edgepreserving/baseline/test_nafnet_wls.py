"""Contract tests for the grayscale WLS NAFNet baseline."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from baseline.wls_data import WLSPairs, collate_pairs, paired_patch_collate, partition_ids
from baseline.train_nafnet_wls import make_model_and_loss


def _write_pair(root: Path, split: str, identifier: str, value: float):
    suffix = "" if split == "train" else "-200"
    input_dir = root / f"{split}input-l0p500-a1p400{suffix}"
    target_dir = root / f"{split}label-l0p500-a1p400{suffix}"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((5, 7), value, dtype=np.float32)
    assert cv2.imwrite(str(input_dir / f"{identifier}-log.tiff"), image)
    assert cv2.imwrite(str(target_dir / f"{identifier}-WLS-l0p500-a1p400.tiff"), image + 0.1)


def test_wls_pairs_match_by_source_id_and_keep_one_channel(tmp_path):
    _write_pair(tmp_path, "train", "b", 0.2)
    _write_pair(tmp_path, "train", "a", 0.4)
    dataset = WLSPairs(tmp_path, "train")
    assert dataset.ids == ["a", "b"]
    image, target = dataset[0]
    assert image.shape == target.shape == (1, 5, 7)
    assert torch.allclose(target - image, torch.full_like(image, 0.1))


def test_wls_pairs_fail_when_a_target_is_missing(tmp_path):
    _write_pair(tmp_path, "train", "a", 0.2)
    (tmp_path / "trainlabel-l0p500-a1p400" / "a-WLS-l0p500-a1p400.tiff").unlink()
    with pytest.raises(ValueError, match="Unmatched"):
        WLSPairs(tmp_path, "train")


def test_train_validation_partition_is_deterministic_and_disjoint():
    identifiers = [f"{i:03d}" for i in range(10)]
    train_a, val_a = partition_ids(identifiers, val_count=2, seed=42)
    train_b, val_b = partition_ids(identifiers, val_count=2, seed=42)
    assert (train_a, val_a) == (train_b, val_b)
    assert len(train_a) == 8 and len(val_a) == 2
    assert set(train_a).isdisjoint(val_a)
    assert sorted(train_a + val_a) == identifiers


def test_patch_training_and_even_full_image_validation_geometry():
    image = torch.arange(1 * 5 * 7, dtype=torch.float32).reshape(1, 5, 7) / 100.0
    target = image + 0.1
    patches, patch_targets = paired_patch_collate([(image, target)], crop_size=4)
    assert patches.shape == (1, 1, 4, 4)
    assert torch.allclose(patch_targets - patches, torch.full_like(patches, 0.1))
    full, full_target = collate_pairs([(image, target)])
    assert full.shape == full_target.shape == (1, 1, 4, 6)


def test_model_uses_one_input_and_output_channel():
    torch.set_num_threads(2)
    model, criterion = make_model_and_loss(torch.device("cpu"))
    assert model.intro.in_channels == 1
    assert model.ending.out_channels == 1
    with torch.inference_mode():
        output = model(torch.rand(1, 1, 32, 32))
    assert output.shape == (1, 1, 32, 32)
    assert criterion.toY is False
