from types import SimpleNamespace

import pytest

import torch

from baseline.train_nafnet_rtv import (
    build_run_protocol,
    paired_patch_collate,
    split_microbatches,
    validate_resume_protocol,
)


def make_args(**overrides):
    values = {
        "batch_size_per_gpu": 8,
        "micro_batch_size": 8,
        "train_crop_size": 256,
        "precision": "bf16",
        "world_size": 3,
        "total_iter": 200000,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "beta1": 0.9,
        "beta2": 0.9,
        "min_lr": 1e-7,
        "val_freq": 20000,
        "save_freq": 5000,
        "seed": 42,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_protocol_keeps_nafnet_recipe_and_full_image_geometry():
    protocol = build_run_protocol(make_args())

    assert protocol["train_geometry"] == "random_paired_patch"
    assert protocol["train_crop_size"] == 256
    assert protocol["validation_geometry"] == "full_image_even"
    assert protocol["effective_batch_size"] == 24
    assert protocol["total_iter"] == 200000
    assert protocol["lr"] == 1e-3
    assert protocol["scheduler"] == "CosineAnnealingLR"
    assert protocol["loss"] == "PSNRLoss"
    assert protocol["pin_memory"] is False
    assert protocol["precision"] == "bf16"


def test_microbatch_split_preserves_full_collated_batch_and_partial_tail():
    images = torch.arange(5 * 3 * 4 * 4).reshape(5, 3, 4, 4)
    targets = images + 1

    pieces = split_microbatches(images, targets, micro_batch_size=2)

    assert [image.shape[0] for image, _ in pieces] == [2, 2, 1]
    assert torch.equal(torch.cat([image for image, _ in pieces]), images)
    assert torch.equal(torch.cat([target for _, target in pieces]), targets)


def test_paired_patch_collate_crops_input_and_target_to_same_random_window():
    torch.manual_seed(2)
    image = torch.arange(3 * 8 * 10).reshape(3, 8, 10)
    target = image + 1000

    images, targets = paired_patch_collate([(image, target)], crop_size=4)

    assert images.shape == (1, 3, 4, 4)
    assert targets.shape == (1, 3, 4, 4)
    assert torch.equal(targets - images, torch.full_like(images, 1000))


def test_paired_patch_collate_reflect_pads_images_smaller_than_crop():
    image = torch.arange(3 * 249 * 260, dtype=torch.float32).reshape(3, 249, 260)
    target = image + 10

    images, targets = paired_patch_collate([(image, target)], crop_size=256)

    assert images.shape == (1, 3, 256, 256)
    assert targets.shape == (1, 3, 256, 256)
    assert torch.equal(targets - images, torch.full_like(images, 10))


def test_resume_rejects_different_effective_batch_or_iteration_budget():
    current = build_run_protocol(make_args())
    changed = build_run_protocol(make_args(batch_size_per_gpu=2))

    with pytest.raises(ValueError, match="protocol does not match"):
        validate_resume_protocol({"run_protocol": changed}, current)


def test_resume_allows_microbatch_change_for_memory_tuning():
    original = build_run_protocol(make_args(micro_batch_size=1))
    tuned = build_run_protocol(make_args(micro_batch_size=4))

    validate_resume_protocol({"run_protocol": original}, tuned)


def test_default_batch_and_microbatch_are_doubled(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_nafnet_rtv.py"])
    from baseline.train_nafnet_rtv import parse_args

    args = parse_args()
    assert args.batch_size_per_gpu == 8
    assert args.micro_batch_size == 8
    assert args.train_crop_size == 256
