from pathlib import Path

import cv2
import numpy as np
import pytest

from exp_b.src.data_exp_b import PairedPatchCollator, PairedTripletDataset


def _write_gray(path: Path, shape):
    image = np.zeros(shape, dtype=np.float32)
    assert cv2.imwrite(str(path), image)


def _write_array(path: Path, image):
    assert cv2.imwrite(str(path), np.asarray(image, dtype=np.float32))


def test_dataset_matches_lr_hr_gt_and_collator_keeps_aligned_crop(tmp_path):
    root = tmp_path / "data"
    for folder in (
        "traininput-l0p500-a1p400_downsample",
        "traininput-l0p500-a1p400",
        "trainlabel-l0p500-a1p400",
    ):
        (root / folder).mkdir(parents=True)
    lr = np.arange(80 * 96, dtype=np.float32).reshape(80, 96)
    hr = np.repeat(np.repeat(lr, 2, axis=0), 2, axis=1)
    _write_array(root / "traininput-l0p500-a1p400_downsample" / "one-log.tif", lr)
    _write_array(root / "traininput-l0p500-a1p400" / "one-log.tif", hr)
    _write_gray(
        root / "trainlabel-l0p500-a1p400" / "one-WLS-l0p500-a1p400.tif",
        (160, 192),
    )

    dataset = PairedTripletDataset(root, split="train")
    lr, hr, gt = dataset[0]
    batch = PairedPatchCollator(crop_size=128)([(lr, hr, gt)])

    assert [tuple(x.shape) for x in batch] == [
        (1, 1, 64, 64),
        (1, 1, 128, 128),
        (1, 1, 128, 128),
    ]
    np.testing.assert_array_equal(
        batch[1][0, 0, ::2, ::2].numpy(), batch[0][0, 0].numpy()
    )


def test_dataset_rejects_missing_or_unmatched_pairs(tmp_path):
    root = tmp_path / "data"
    (root / "traininput-l0p500-a1p400_downsample").mkdir(parents=True)
    (root / "traininput-l0p500-a1p400").mkdir(parents=True)
    (root / "trainlabel-l0p500-a1p400").mkdir(parents=True)
    _write_gray(root / "traininput-l0p500-a1p400_downsample" / "one.tif", (16, 16))
    _write_gray(root / "traininput-l0p500-a1p400" / "one.tif", (32, 32))
    _write_gray(root / "trainlabel-l0p500-a1p400" / "other.tif", (32, 32))

    with pytest.raises(ValueError, match="match"):
        PairedTripletDataset(root, split="train")
