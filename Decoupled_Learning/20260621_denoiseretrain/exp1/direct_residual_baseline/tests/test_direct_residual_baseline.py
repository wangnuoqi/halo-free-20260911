import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from data import build_datasets
from loss import cascade_loss
from model import DirectResidual3Stage


class DirectResidualBaselineTests(unittest.TestCase):
    def test_build_datasets_uses_all_train_images_and_separate_bsd68_root(self):
        image = np.arange(7 * 9, dtype=np.uint8).reshape(7, 9)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_root = root / "BSDS400"
            validation_root = root / "BSD68"
            train_root.mkdir()
            validation_root.mkdir()
            for index in range(2):
                self.assertTrue(cv2.imwrite(str(train_root / f"train_{index}.png"), image))
            self.assertTrue(cv2.imwrite(str(validation_root / "validation.png"), image))

            train_dataset, validation_dataset = build_datasets(
                train_root, validation_root, sigma=25.0
            )

        self.assertEqual(len(train_dataset), 2)
        self.assertEqual(len(validation_dataset), 1)
        self.assertTrue(train_dataset.augment)
        self.assertFalse(validation_dataset.augment)
        self.assertEqual(validation_dataset.deterministic_noise_seed, 0)

    def test_zero_initialized_model_preserves_noisy_baseline(self):
        model = DirectResidual3Stage().eval()
        for parameter in model.parameters():
            parameter.data.zero_()
        noisy = torch.rand(1, 1, 16, 20)

        output = model(noisy)["final"]

        self.assertTrue(torch.allclose(output, noisy, atol=1e-6))

    def test_cascade_loss_is_zero_for_exact_noise_residuals(self):
        noisy = torch.rand(1, 1, 16, 20)
        clean = torch.rand(1, 1, 16, 20)
        target_delta = clean - noisy
        outputs = {
            "delta1": target_delta,
            "delta2": target_delta,
            "delta3": target_delta,
        }

        loss, stage_losses = cascade_loss(outputs, noisy, clean)

        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        self.assertTrue(all(abs(stage_loss.item()) < 1e-6 for stage_loss in stage_losses))


if __name__ == "__main__":
    unittest.main()
