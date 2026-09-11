import unittest
from pathlib import Path

import numpy as np

from sobel_visualize_compare import (
    common_prefix_id,
    compute_sobel_magnitude,
    dilate_mask,
    normalize_with_limits,
    robust_limits,
)


class TestSobelVisualizeCompare(unittest.TestCase):
    def test_common_prefix_id_strips_known_suffixes(self):
        cases = {
            "202409121511100_NormalizedHDR-log_pred.tiff": "202409121511100_NormalizedHDR",
            "202409121511100_NormalizedHDR-log.tiff": "202409121511100_NormalizedHDR",
            "202409121511100_NormalizedHDR-WLS-025000.tiff": "202409121511100_NormalizedHDR",
        }
        for name, expected in cases.items():
            self.assertEqual(common_prefix_id(Path(name)), expected)

    def test_sobel_magnitude_detects_vertical_step(self):
        img = np.zeros((9, 9), dtype=np.float32)
        img[:, 5:] = 10.0

        grad = compute_sobel_magnitude(img)

        self.assertEqual(grad.shape, img.shape)
        self.assertGreater(float(grad[:, 4:6].mean()), float(grad[:, :2].mean()) + 1.0)

    def test_robust_limits_share_one_scale(self):
        a = np.array([0, 1, 2], dtype=np.float32)
        b = np.array([0, 10, 100], dtype=np.float32)

        vmin, vmax = robust_limits([a, b], low=0, high=100)

        self.assertEqual(vmin, 0.0)
        self.assertEqual(vmax, 100.0)

    def test_normalize_with_shared_limits(self):
        img = np.array([-1, 0, 5, 10, 20], dtype=np.float32)

        scaled = normalize_with_limits(img, 0, 10)

        np.testing.assert_allclose(scaled, np.array([0, 0, 0.5, 1, 1], dtype=np.float32))

    def test_dilate_mask_expands_edge_band(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True

        band = dilate_mask(mask, radius=1)

        self.assertEqual(int(band.sum()), 9)
        self.assertTrue(band[1, 1])
        self.assertTrue(band[3, 3])
        self.assertFalse(band[0, 0])


if __name__ == "__main__":
    unittest.main()
