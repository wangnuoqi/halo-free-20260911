import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from eval_rf_3stagev3 import (
    CascadeRefine3Stage,
    GuidedGrayRefineEvalDataset,
    calc_psnr,
    calc_ssim,
    load_state_dict_flexible,
    normalize_path_for_runtime,
    output_dir_for_sigma,
    save_gray_prediction,
    var_custom_collate_with_names,
)


def write_gray(path: Path, value: int, shape=(4, 5)) -> None:
    image = np.full(shape, value, dtype=np.uint8)
    imageio.imwrite(path, image)


class EvalRF3StageV3Tests(unittest.TestCase):
    def test_output_dir_for_sigma_keeps_rf_results_separate(self):
        root = Path("/tmp/experiment")
        self.assertEqual(
            output_dir_for_sigma(root, 25),
            root / "eval_results-img-rf" / "sigma25",
        )

    def test_normalize_path_for_runtime_converts_windows_path_on_posix(self):
        converted = normalize_path_for_runtime(r"D:\nuoqi\halo-free\data")

        self.assertEqual(converted, "/mnt/d/nuoqi/halo-free/data")

    def test_dataset_matches_shared_sorted_gray_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            coarse_dir = root / "coarse"
            label_dir = root / "label"
            raw_dir.mkdir()
            coarse_dir.mkdir()
            label_dir.mkdir()
            write_gray(raw_dir / "b.png", 10)
            write_gray(raw_dir / "a.png", 20)
            write_gray(coarse_dir / "b.png", 30)
            write_gray(coarse_dir / "a.png", 40)
            write_gray(label_dir / "b.png", 50)
            write_gray(label_dir / "a.png", 60)

            dataset = GuidedGrayRefineEvalDataset(raw_dir, coarse_dir, label_dir, sigma=0)

            self.assertEqual(len(dataset), 2)
            sample = dataset[0]
            self.assertEqual(sample["filename"], "a.png")
            self.assertEqual(tuple(sample["origin_input_hr"].shape), (1, 4, 5))
            self.assertEqual(tuple(sample["input_hr"].shape), (1, 4, 5))
            self.assertEqual(tuple(sample["target_img"].shape), (1, 4, 5))

    def test_collate_crops_to_minimum_size_and_keeps_names(self):
        batch = [
            {
                "origin_input_hr": torch.ones(1, 5, 6),
                "input_hr": torch.ones(1, 5, 6) * 2,
                "target_img": torch.ones(1, 5, 6) * 3,
                "filename": "a",
                "input_height_hr": 5,
                "input_width_hr": 6,
            },
            {
                "origin_input_hr": torch.ones(1, 4, 7),
                "input_hr": torch.ones(1, 4, 7) * 4,
                "target_img": torch.ones(1, 4, 7) * 5,
                "filename": "b",
                "input_height_hr": 4,
                "input_width_hr": 7,
            },
        ]

        raw, coarse, label, names, was_rotated = var_custom_collate_with_names(batch)

        self.assertEqual(tuple(raw.shape), (2, 1, 4, 6))
        self.assertEqual(tuple(coarse.shape), (2, 1, 4, 6))
        self.assertEqual(tuple(label.shape), (2, 1, 4, 6))
        self.assertEqual(names, ["a", "b"])
        self.assertEqual(was_rotated, [False, False])

    def test_load_state_dict_flexible_accepts_module_prefix_mismatch(self):
        plain_model = torch.nn.Conv2d(3, 3, kernel_size=1)
        wrapped_model = torch.nn.DataParallel(torch.nn.Conv2d(3, 3, kernel_size=1))

        load_state_dict_flexible(plain_model, wrapped_model.state_dict())

        self.assertTrue(torch.allclose(plain_model.weight, wrapped_model.module.weight))

    def test_save_gray_prediction_writes_uint8_gray_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.png"
            prediction = torch.ones(1, 2, 3) * 0.5

            save_gray_prediction(prediction, path)

            loaded = imageio.imread(path)
            self.assertEqual(loaded.dtype, np.uint8)
            self.assertEqual(tuple(loaded.shape), (2, 3))

    def test_calc_psnr_reports_expected_values(self):
        pred = torch.zeros(1, 1, 2, 2)
        target = torch.ones(1, 1, 2, 2) * 0.5

        psnr, mse = calc_psnr(pred, target)

        self.assertAlmostEqual(mse, 0.25)
        self.assertAlmostEqual(psnr, 6.020599913279624)

    def test_calc_ssim_is_one_for_identical_images_and_lower_for_different_images(self):
        image = torch.rand(2, 1, 16, 16)
        different = torch.clamp(image * 0.5, 0.0, 1.0)

        identical_ssim = calc_ssim(image, image)
        different_ssim = calc_ssim(different, image)

        self.assertAlmostEqual(identical_ssim, 1.0, places=5)
        self.assertLess(different_ssim, identical_ssim)


if __name__ == "__main__":
    unittest.main()
