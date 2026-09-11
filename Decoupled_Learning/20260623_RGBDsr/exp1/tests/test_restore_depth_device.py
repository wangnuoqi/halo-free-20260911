"""Regression coverage for CPU metadata used with GPU predictions."""

from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "test_convguided_base.py"


def test_restore_depth_moves_both_depth_bounds_before_subtraction():
    body = SOURCE.read_text(encoding="utf-8")
    start = body.index("def restore_depth")
    end = body.index("\ndef compute_image_metrics", start)
    function = body[start:end]

    assert "depth_max = depth_max.to(device=normalized.device, dtype=normalized.dtype)" in function
    assert function.index("depth_max = depth_max.to") < function.index("depth_max - depth_min.view(-1)")
