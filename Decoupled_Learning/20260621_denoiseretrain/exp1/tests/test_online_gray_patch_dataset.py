import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from online_gray_patch_dataset import OnlineGrayPatchDataset


def _write_gray_jpg(path):
    image = np.arange(96 * 96, dtype=np.uint8).reshape(96, 96)
    assert cv2.imwrite(str(path), image)


def test_dataset_returns_gray_aligned_patches(tmp_path):
    _write_gray_jpg(tmp_path / "sample.jpg")
    dataset = OnlineGrayPatchDataset(tmp_path, sigma=0, scales=(1.0,), train=False)
    lr, noisy_hr, clean_hr = dataset[0]
    assert lr.shape == (1, 20, 20)
    assert noisy_hr.shape == (1, 40, 40)
    assert clean_hr.shape == (1, 40, 40)
    assert lr.dtype == noisy_hr.dtype == clean_hr.dtype == torch.float32
    assert torch.allclose(noisy_hr, clean_hr)


def test_eval_noise_is_deterministic(tmp_path):
    _write_gray_jpg(tmp_path / "sample.jpg")
    dataset = OnlineGrayPatchDataset(tmp_path, sigma=25, scales=(1.0,), train=False)
    first = dataset[0]
    second = dataset[0]
    assert all(torch.equal(a, b) for a, b in zip(first, second))
