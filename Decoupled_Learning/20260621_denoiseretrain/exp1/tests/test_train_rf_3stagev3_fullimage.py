import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from train_rf_3stagev3_fullimage import (
    FullImageGrayDenoisingDataset,
    build_datasets,
)


class FullImageRefineDatasetTests(unittest.TestCase):
    def test_build_datasets_uses_all_training_images_and_separate_validation_root(self):
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
                train_root,
                validation_root,
                sigma=25.0,
            )

        self.assertEqual(len(train_dataset), 2)
        self.assertEqual(len(validation_dataset), 1)
        self.assertTrue(train_dataset.augment)
        self.assertFalse(validation_dataset.augment)
        self.assertEqual(validation_dataset.deterministic_noise_seed, 0)

    def test_full_image_sample_keeps_noisy_target_and_lr_aligned(self):
        image = np.arange(7 * 9, dtype=np.uint8).reshape(7, 9)
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            self.assertTrue(cv2.imwrite(str(image_path), image))
            dataset = FullImageGrayDenoisingDataset(
                [image_path],
                sigma=0.0,
                scale=2,
                augment=True,
            )

            sample = dataset[0]

        clean = sample["clean_hr"]
        noisy = sample["noisy_hr"]
        noisy_lr = sample["noisy_lr"]
        self.assertIn(tuple(clean.shape), {(1, 6, 8), (1, 8, 6)})
        self.assertTrue(torch.equal(noisy, clean))
        expected_height, expected_width = clean.shape[-2] // 2, clean.shape[-1] // 2
        self.assertEqual(tuple(noisy_lr.shape), (1, expected_height, expected_width))
        expected_lr = cv2.resize(
            clean[0].numpy(),
            (expected_width, expected_height),
            interpolation=cv2.INTER_LANCZOS4,
        )
        np.testing.assert_allclose(noisy_lr[0].numpy(), expected_lr, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
