from types import SimpleNamespace

import pytest

from baseline.train_rtv import build_run_protocol, validate_resume_protocol


def test_full_image_protocol_is_distinguished_from_fixed_crop_protocol():
    full_image = build_run_protocol(SimpleNamespace(
        crop_size=None, eval_crop_size=None, batch_size=8
    ))
    crop = build_run_protocol(SimpleNamespace(
        crop_size=256, eval_crop_size=256, batch_size=8
    ))

    assert full_image["train_geometry"] == "batch_min_top_left_even"
    assert full_image["validation_geometry"] == "full_image_even"
    assert crop["train_geometry"] == "random_crop"
    assert crop["validation_geometry"] == "center_crop"
    assert full_image != crop


def test_resume_rejects_checkpoint_from_a_different_geometry():
    current = build_run_protocol(SimpleNamespace(
        crop_size=None, eval_crop_size=None, batch_size=8
    ))
    old_crop = build_run_protocol(SimpleNamespace(
        crop_size=256, eval_crop_size=256, batch_size=8
    ))

    with pytest.raises(ValueError, match="protocol does not match"):
        validate_resume_protocol({"run_protocol": old_crop}, current)
