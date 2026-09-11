import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from dncnn_gray_lr_patch_dataset import DnCNNGrayLRPatchDataset


def _write_gray_jpg(path):
    image = np.arange(96 * 96, dtype=np.uint8).reshape(96, 96)
    assert cv2.imwrite(str(path), image)


def test_train_returns_dncnn_hr_patch_and_aligned_lr(tmp_path):
    _write_gray_jpg(tmp_path / "sample.jpg")
    dataset = DnCNNGrayLRPatchDataset(
        tmp_path, sigma=0, patch_size=40, scale=2, train=True, seed=42
    )
    lr, noisy_hr, clean_hr = dataset[0]
    assert lr.shape == (1, 20, 20)
    assert noisy_hr.shape == (1, 40, 40)
    assert clean_hr.shape == (1, 40, 40)
    assert lr.dtype == noisy_hr.dtype == clean_hr.dtype == torch.float32
    assert torch.equal(noisy_hr, clean_hr)
    expected_lr = cv2.resize(
        clean_hr[0].numpy(), (20, 20), interpolation=cv2.INTER_LANCZOS4
    )
    assert np.allclose(lr[0].numpy(), expected_lr)


def test_train_default_patch_covers_guided_filter_context(tmp_path):
    image = np.arange(321 * 481, dtype=np.uint32).reshape(321, 481) % 256
    assert cv2.imwrite(str(tmp_path / "large.jpg"), image.astype(np.uint8))

    dataset = DnCNNGrayLRPatchDataset(tmp_path, sigma=0, scale=2, train=True)
    lr, noisy_hr, clean_hr = dataset[0]

    assert lr.shape == (1, 96, 96)
    assert noisy_hr.shape == (1, 192, 192)
    assert clean_hr.shape == (1, 192, 192)


def test_eval_is_deterministic_and_keeps_full_image(tmp_path):
    _write_gray_jpg(tmp_path / "sample.jpg")
    dataset = DnCNNGrayLRPatchDataset(tmp_path, sigma=25, scale=2, train=False)
    first = dataset[0]
    second = dataset[0]
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert first[1].shape == (1, 96, 96)
    assert first[0].shape == (1, 48, 48)
    assert first[2].shape == (1, 96, 96)
