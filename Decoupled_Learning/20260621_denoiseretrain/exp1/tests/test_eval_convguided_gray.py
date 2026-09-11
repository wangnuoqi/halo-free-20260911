import unittest
import tempfile
from pathlib import Path

import cv2
import torch

from eval_convguided_gray import metrics_path_for, output_dir_for_sigma, save_prediction


class MetricsPathTests(unittest.TestCase):
    def test_metrics_path_is_unique_per_sigma(self):
        root = Path("/tmp/eval_result_gray")
        self.assertEqual(
            metrics_path_for(root, 15),
            root / "metrics_sigma15.json",
        )
        self.assertEqual(
            metrics_path_for(root, 25.0),
            root / "metrics_sigma25.json",
        )
        self.assertEqual(
            metrics_path_for(root, 12.5),
            root / "metrics_sigma12.5.json",
        )

    def test_output_dir_for_sigma_uses_sigma_subdirectory(self):
        root = Path("/tmp/eval_results-img")
        self.assertEqual(output_dir_for_sigma(root, 15), root / "sigma15")
        self.assertEqual(output_dir_for_sigma(root, 25.0), root / "sigma25")
        self.assertEqual(output_dir_for_sigma(root, 12.5), root / "sigma12.5")

    def test_save_prediction_writes_grayscale_png(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sigma25" / "test001.png"
            save_prediction(torch.tensor([[[0.0, 0.5], [1.0, 0.25]]]), path)
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            self.assertEqual(image.shape, (2, 2))
            self.assertEqual(image.dtype, "uint8")
            self.assertEqual(image.tolist(), [[0, 128], [255, 64]])


if __name__ == "__main__":
    unittest.main()
