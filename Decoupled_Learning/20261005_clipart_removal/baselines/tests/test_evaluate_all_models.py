import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_all_models


class EvaluateAllModelsTests(unittest.TestCase):
    def test_joint_checkpoint_override_loads_patch64_run(self):
        checkpoint = PROJECT_ROOT / "train_clipart_joint/ckpt/exp3-b_joint_patch128_noaug_bs64/joint_best.pth"
        model = evaluate_all_models.load_ours(
            "exp3-b", PROJECT_ROOT, torch.device("cpu"), checkpoint_override=checkpoint
        )
        self.assertEqual(model["checkpoint"], checkpoint)
        self.assertEqual(model["channels"], 3)

    def test_fbcnn_quality_input_matches_image_batch(self):
        class BatchCheckingFBCNN:
            def __init__(self):
                self.quality_batch_size = None

            def __call__(self, images, quality):
                self.quality_batch_size = quality.shape[0]
                if self.quality_batch_size != images.shape[0]:
                    raise AssertionError("FBCNN quality batch must match image batch")
                return images, quality

        with tempfile.TemporaryDirectory() as temp_dir:
            image_paths = []
            for name in ("first.png", "second.png"):
                path = Path(temp_dir) / name
                Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
                image_paths.append(path)

            model = BatchCheckingFBCNN()
            predictions = evaluate_all_models.predict_baseline(
                "FBCNN", model, image_paths, torch.device("cpu")
            )

        self.assertEqual(model.quality_batch_size, 2)
        self.assertEqual(len(predictions), 2)

    def test_rgb_half_resolution_matches_training_lanczos(self):
        image = np.zeros((8, 10, 3), dtype=np.float32)
        image[2:7, 3:9] = 0.75
        actual = evaluate_all_models.resize_half_rgb(image)
        source = Image.fromarray((image * 255).astype(np.uint8))
        expected = np.asarray(
            source.resize((5, 4), Image.Resampling.LANCZOS), dtype=np.float32
        ) / 255.0
        np.testing.assert_array_equal(actual, expected)

    def test_swinir_protocol_checkpoint_uses_training_patch_window_geometry(self):
        models = evaluate_all_models.load_baselines(
            PROJECT_ROOT, torch.device("cpu"), ["SwinIR"]
        )
        self.assertIn("SwinIR", models)

    def test_vivnet_protocol_checkpoint_runs_full_image_inference(self):
        device = torch.device("cpu")
        models = evaluate_all_models.load_baselines(PROJECT_ROOT, device, ["VIVNet"])
        self.assertIn("VIVNet", models)

        test_list = PROJECT_ROOT / "openclipart_5000" / "test_list.txt"
        stem = test_list.read_text(encoding="utf-8").splitlines()[0]
        image_path = PROJECT_ROOT / "openclipart_5000" / "jpeg_pairs_512" / "jpeg_q30" / f"{stem}.jpg"
        predictions = evaluate_all_models.predict_baseline("VIVNet", models["VIVNet"], [image_path], device)

        self.assertEqual(predictions[0].shape, (512, 512, 3))
        self.assertTrue(np.isfinite(predictions[0]).all())


if __name__ == "__main__":
    unittest.main()
