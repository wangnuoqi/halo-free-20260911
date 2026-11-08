import numpy as np
import pytest
import torch
import tifffile

from baseline.rtv_data import RTVPairDataset, _read_rgb_255, build_model_input, encode_rtv_condition, paired_crop_collate


def test_dataset_pairs_by_stem_and_scales_tiff_to_stage_channel_order(tmp_path):
    input_dir = tmp_path / "train" / "input"
    label_dir = tmp_path / "train" / "label"
    input_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    tifffile.imwrite(input_dir / "image_a.tiff", np.full((32, 40, 3), 0.25, np.float32), photometric="rgb")
    tifffile.imwrite(input_dir / "image_b.tiff", np.full((32, 40, 3), 0.75, np.float32), photometric="rgb")
    tifffile.imwrite(label_dir / "image_a.tiff", np.full((32, 40, 3), 0.50, np.float32), photometric="rgb")
    tifffile.imwrite(label_dir / "image_b.tiff", np.full((32, 40, 3), 0.10, np.float32), photometric="rgb")

    dataset = RTVPairDataset(tmp_path / "train")
    image, label = dataset[0]

    assert len(dataset) == 2
    assert image.shape == (3, 32, 40)
    assert label.shape == (3, 32, 40)
    assert torch.allclose(image, torch.full_like(image, 63.75))
    assert torch.allclose(label, torch.full_like(label, 127.5))


def test_tiff_reader_uses_same_bgr_channel_order_as_three_stage_loader(tmp_path):
    path = tmp_path / "rgb_channels.tiff"
    image = np.zeros((4, 5, 3), dtype=np.float32)
    image[..., 0] = 0.1
    image[..., 1] = 0.5
    image[..., 2] = 0.9
    tifffile.imwrite(path, image, photometric="rgb")

    loaded = _read_rgb_255(path)

    assert torch.allclose(loaded[:, 0, 0], torch.tensor([0.9, 0.5, 0.1]) * 255)


def test_model_input_has_rgb_plus_edge_and_official_centering():
    image = torch.zeros(2, 3, 16, 16)
    image[:, :, :, 8:] = 255

    model_input = build_model_input(image)

    assert model_input.shape == (2, 4, 16, 16)
    assert torch.equal(model_input[:, :3], image - 128)
    assert torch.all(model_input[:, 3] < 0)
    assert torch.any(model_input[:, 3] > -128)


def test_rtv_lambda_uses_official_single_operator_condition_mapping():
    assert encode_rtv_condition(0.01) == pytest.approx(0.035)


def test_paired_crop_collate_keeps_pairs_aligned_for_different_image_sizes():
    image_a = torch.arange(3 * 20 * 24, dtype=torch.float32).reshape(3, 20, 24)
    label_a = image_a + 10
    image_b = torch.arange(3 * 28 * 30, dtype=torch.float32).reshape(3, 28, 30)
    label_b = image_b + 10

    images, labels = paired_crop_collate(
        [(image_a, label_a), (image_b, label_b)], crop_size=16, random_crop=False
    )

    assert images.shape == labels.shape == (2, 3, 16, 16)
    assert torch.equal(labels, images + 10)


def test_paired_crop_collate_full_image_mode_keeps_entire_even_sized_pair():
    image = torch.arange(3 * 20 * 24, dtype=torch.float32).reshape(3, 20, 24)
    label = image + 10

    images, labels = paired_crop_collate([(image, label)], crop_size=None, random_crop=True)

    assert images.shape == labels.shape == (1, 3, 20, 24)
    assert torch.equal(images[0], image)
    assert torch.equal(labels[0], label)


def test_paired_crop_collate_full_image_mode_trims_odd_border_like_stage_data():
    image = torch.arange(3 * 21 * 25, dtype=torch.float32).reshape(3, 21, 25)
    label = image + 10

    images, labels = paired_crop_collate([(image, label)], crop_size=None, random_crop=False)

    assert images.shape == labels.shape == (1, 3, 20, 24)
    assert torch.equal(labels, images + 10)


def test_paired_crop_collate_full_image_mode_matches_three_stage_batch_geometry():
    image_a = torch.arange(3 * 20 * 24, dtype=torch.float32).reshape(3, 20, 24)
    label_a = image_a + 10
    image_b = torch.arange(3 * 23 * 27, dtype=torch.float32).reshape(3, 23, 27)
    label_b = image_b + 10

    images, labels = paired_crop_collate(
        [(image_a, label_a), (image_b, label_b)], crop_size=None, random_crop=True
    )

    assert images.shape == labels.shape == (2, 3, 20, 24)
    assert torch.equal(images[0], image_a)
    assert torch.equal(images[1], image_b[:, :20, :24])
    assert torch.equal(labels, images + 10)
