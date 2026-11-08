import math

import numpy as np
import pytest

from exp_b.src.evaluate_exp_b import image_metrics, trim_even


def test_metrics_use_per_image_mse_and_unit_range_psnr():
    target = np.zeros((3, 5), dtype=np.float32)
    prediction = np.ones((3, 5), dtype=np.float32) * 0.1
    metrics = image_metrics(prediction, target)
    assert metrics["mse"] == pytest.approx(0.01, abs=1e-7)
    assert metrics["psnr_db"] == pytest.approx(20.0, abs=1e-5)


def test_trim_even_removes_only_bottom_and_right_border():
    image = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    trimmed = trim_even(image)
    assert trimmed.shape == (4, 6)
    np.testing.assert_array_equal(trimmed, image[:4, :6])


def test_report_crop_is_centered_512_and_keeps_full_crop():
    from exp_b.src.evaluate_exp_b import report_crop_512

    image = np.arange(700 * 800, dtype=np.int64).reshape(700, 800)
    cropped = report_crop_512(image)
    assert cropped.shape == (512, 512)
    np.testing.assert_array_equal(cropped, image[94:606, 144:656])


def test_report_crop_512_rejects_smaller_images():
    from exp_b.src.evaluate_exp_b import report_crop_512

    with pytest.raises(ValueError, match='at least 512'):
        report_crop_512(np.zeros((500, 600), dtype=np.float32))


def test_report_crop_triplet_preserves_aligned_2x_geometry():
    from exp_b.src.evaluate_exp_b import report_crop_triplet

    lr = np.arange(300 * 400, dtype=np.int64).reshape(1, 300, 400)
    hr = np.arange(600 * 800, dtype=np.int64).reshape(1, 600, 800)
    gt = hr + 1
    lr_crop, hr_crop, gt_crop = report_crop_triplet(lr, hr, gt)
    assert lr_crop.shape == (1, 256, 256)
    assert hr_crop.shape == gt_crop.shape == (1, 512, 512)
    np.testing.assert_array_equal(hr_crop, hr[:, 44:556, 144:656])
    np.testing.assert_array_equal(lr_crop, lr[:, 22:278, 72:328])
