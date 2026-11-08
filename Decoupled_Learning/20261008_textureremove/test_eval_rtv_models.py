"""Focused tests for the shared RTV test protocol."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from eval_rtv_models import discover_pairs, ensure_empty_output, load_pair, score_prediction, summarize_rows, write_preview


def test_discover_pairs_requires_exactly_matching_names(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "label").mkdir()
    (tmp_path / "input" / "a.tiff").touch()
    (tmp_path / "label" / "b.tiff").touch()
    with pytest.raises(ValueError, match="Unmatched"):
        discover_pairs(tmp_path)


def test_load_pair_uses_shared_bgr_and_even_top_left_crop(tmp_path):
    image = np.zeros((5, 7, 3), dtype=np.float32)
    image[..., 0], image[..., 1], image[..., 2] = 0.1, 0.3, 0.7
    target = image + 0.1
    input_path, target_path = tmp_path / "input.tiff", tmp_path / "label.tiff"
    assert cv2.imwrite(str(input_path), image)
    assert cv2.imwrite(str(target_path), target)
    loaded_image, loaded_target = load_pair(input_path, target_path)
    assert loaded_image.shape == loaded_target.shape == (1, 3, 4, 6)
    assert torch.allclose(loaded_image[0, :, 0, 0], torch.tensor([0.1, 0.3, 0.7]))
    assert torch.allclose(loaded_target - loaded_image, torch.full_like(loaded_image, 0.1))


def test_score_prediction_uses_clamped_zero_to_one_output():
    prediction = torch.tensor([[[[1.2, -0.2]]]])
    target = torch.tensor([[[[0.5, 0.5]]]])
    mse, mae = score_prediction(prediction, target)
    assert mse == pytest.approx(0.25)
    assert mae == pytest.approx(0.5)


def test_summary_is_mean_per_image_and_reports_time_distribution():
    rows = [
        {"mse": 0.1, "mae": 0.2, "infer_ms": 10.0},
        {"mse": 0.3, "mae": 0.4, "infer_ms": 30.0},
    ]
    result = summarize_rows(rows)
    assert result["images"] == 2
    assert result["mse"] == pytest.approx(0.2)
    assert result["mae"] == pytest.approx(0.3)
    assert result["mean_infer_ms"] == pytest.approx(20.0)
    assert result["p50_infer_ms"] == pytest.approx(20.0)


def test_preview_places_input_target_and_model_outputs_side_by_side(tmp_path):
    for name in ("input", "target", "stage1"):
        folder = tmp_path / "images" / name
        folder.mkdir(parents=True)
        assert cv2.imwrite(str(folder / "sample.png"), np.zeros((4, 6, 3), dtype=np.uint8))
    output = write_preview(tmp_path, "sample", ["stage1"])
    image = cv2.imread(str(output))
    assert image.shape == (36, 18, 3)


def test_nonempty_result_directory_is_rejected_to_avoid_stale_files(tmp_path):
    (tmp_path / "old.csv").write_text("old result")
    with pytest.raises(FileExistsError, match="not empty"):
        ensure_empty_output(tmp_path)
