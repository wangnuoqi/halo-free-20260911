"""Ensure Conv checkpoints can resume across single- and multi-GPU runs."""

from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "train_convguided_base.py"


def test_training_script_has_checkpoint_prefix_compatibility_loader():
    source = SOURCE.read_text(encoding="utf-8")
    assert "def load_model_state" in source
    assert 'f"module.{key}"' in source
    assert "key[7:]" in source
