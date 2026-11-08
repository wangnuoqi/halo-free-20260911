from pathlib import Path

import numpy as np
from PIL import Image

from tools.build_refine_dataset import (
    fit_to_target,
    make_synthetic_image,
    make_synthetic_record,
    progress_iter,
    resize_target_pair,
)


def test_fit_to_target_preserves_rgb_and_exact_dimensions():
    reflection = Image.new("RGB", (11, 7), (10, 20, 30))

    fitted = fit_to_target(reflection, (5, 9))

    assert fitted.mode == "RGB"
    assert fitted.size == (9, 5)


def test_synthetic_image_matches_target_dimensions_and_range():
    target = Image.new("RGB", (8, 6), (120, 100, 80))
    reflection = Image.new("RGB", (4, 3), (30, 50, 70))

    blended = make_synthetic_image(target, reflection, alpha=0.9, beta=0.4)
    pixels = np.asarray(blended)

    assert blended.size == target.size
    assert blended.mode == "RGB"
    assert pixels.dtype == np.uint8
    assert pixels.min() >= 0
    assert pixels.max() <= 255


def test_synthetic_record_is_traceable(tmp_path: Path):
    record = make_synthetic_record(
        sample_id="synthetic_000001",
        input_path=tmp_path / "blended.png",
        target_path=tmp_path / "transmission.png",
        source_t=tmp_path / "t.png",
        source_r=tmp_path / "r.png",
        alpha=0.9,
        beta=0.4,
        seed=123,
        width=8,
        height=6,
    )

    assert record["id"] == "synthetic_000001"
    assert record["type"] == "synthetic"
    assert record["input"] == str(tmp_path / "blended.png")
    assert record["target"] == str(tmp_path / "transmission.png")
    assert record["source_t"] == str(tmp_path / "t.png")
    assert record["source_r"] == str(tmp_path / "r.png")
    assert record["width"] == 8
    assert record["height"] == 6
    assert record["seed"] == 123


def test_progress_iter_passes_total_and_description(monkeypatch):
    import tools.build_refine_dataset as module

    calls = {}

    def fake_tqdm(iterable, **kwargs):
        calls["iterable"] = iterable
        calls["kwargs"] = kwargs
        return iterable

    monkeypatch.setattr(module, "tqdm", fake_tqdm)

    values = list(progress_iter(range(3), total=3, desc="Generating"))

    assert values == [0, 1, 2]
    assert calls["kwargs"] == {"total": 3, "desc": "Generating", "unit": "sample"}


def test_resize_target_pair_returns_fixed_square_images():
    target = Image.new("RGB", (640, 428), (120, 100, 80))
    reflection = Image.new("RGB", (306, 425), (30, 50, 70))

    resized_target, resized_reflection = resize_target_pair(target, reflection, 768)

    assert resized_target.size == (768, 768)
    assert resized_reflection.size == (768, 768)
