import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "train_clipart_joint"))
from clipart_conv_base_common import make_dataset


class JointPatchDataTests(unittest.TestCase):
    def test_paired_train_crop_then_lanczos_downsample_and_full_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "jpeg_q30").mkdir()
            (root / "clean_png").mkdir()
            pixels = np.random.default_rng(42).integers(0, 256, (512, 512, 3), dtype=np.uint8)
            # Lossless data under both names makes exact pair alignment testable.
            Image.fromarray(pixels).save(root / "jpeg_q30/sample.jpg", format="PNG")
            Image.fromarray(pixels).save(root / "clean_png/sample.png")
            for split in ("train", "val"):
                (root / f"{split}_list.txt").write_text("sample\n")
            train = make_dataset(root, 30, "train", root, 3, patch_size=128)
            lr, hr, gt = train[0]
            self.assertEqual(tuple(hr.shape), (3, 128, 128))
            self.assertEqual(tuple(lr.shape), (3, 64, 64))
            torch.testing.assert_close(hr, gt, rtol=0, atol=0)
            source = Image.fromarray((hr.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            expected = np.asarray(source.resize((64, 64), Image.Resampling.LANCZOS)).transpose(2, 0, 1) / 255.0
            np.testing.assert_allclose(lr.numpy(), expected, rtol=1e-6, atol=1e-7)
            val = make_dataset(root, 30, "val", root, 3, patch_size=128)
            val_lr, val_hr, _ = val[0]
            self.assertEqual(tuple(val_hr.shape), (3, 512, 512))
            self.assertEqual(tuple(val_lr.shape), (3, 256, 256))
            legacy = make_dataset(root, 30, "train", root, 3)
            self.assertEqual(tuple(legacy[0][1].shape), (3, 512, 512))


if __name__ == "__main__":
    unittest.main()
