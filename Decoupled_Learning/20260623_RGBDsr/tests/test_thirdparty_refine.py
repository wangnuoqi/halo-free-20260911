import numpy as np
import torch
from PIL import Image

from thirdparty_refine_models import (
    build_refine,
    ThirdPartyCoarse,
    _pil_bicubic_upsample,
    _postprocess_coarse,
    _prepare_dkn_rgb,
    _prepare_dctnet_rgb,
)


def test_both_refine_variants_preserve_full_image_shape():
    x = torch.rand(1, 3, 32, 40)
    coarse = torch.rand(1, 1, 32, 40)
    for variant in ("original", "exp4b"):
        model = build_refine(variant)
        output = model(x, coarse)["final"]
        assert output.shape == coarse.shape


def test_coarse_adapter_names_are_explicit():
    assert ThirdPartyCoarse.supported_methods() == ("dkn", "dctnet")


def test_pil_bicubic_upsample_matches_pillow_reference():
    low = torch.tensor([[[[0.0, 0.25], [0.75, 1.0]]]], dtype=torch.float32)
    result = _pil_bicubic_upsample(low, (5, 7))
    reference = np.asarray(
        Image.fromarray(low[0, 0].numpy()).resize((7, 5), Image.Resampling.BICUBIC),
        dtype=np.float32,
    ).copy()
    assert result.shape == (1, 1, 5, 7)
    torch.testing.assert_close(result[0, 0], torch.from_numpy(reference))


def test_coarse_postprocess_can_preserve_unclipped_values():
    coarse = torch.tensor([[[[-0.2, 0.4], [1.2, 0.8]]]])
    torch.testing.assert_close(_postprocess_coarse(coarse, clamp=True), coarse.clamp(0.0, 1.0))
    torch.testing.assert_close(_postprocess_coarse(coarse, clamp=False), coarse)


def test_dkn_rgb_order_switch_can_preserve_rgb_channels():
    rgb = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    torch.testing.assert_close(_prepare_dkn_rgb(rgb, bgr=True), rgb[:, [2, 1, 0]])
    torch.testing.assert_close(_prepare_dkn_rgb(rgb, bgr=False), rgb)


def test_dctnet_rgb_preparation_matches_official_normalization():
    rgb = torch.tensor([[[[0.5]], [[0.5]], [[0.5]]]])
    expected = (rgb - rgb.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / rgb.new_tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)
    torch.testing.assert_close(_prepare_dctnet_rgb(rgb), expected)


def test_refine_can_backpropagate_through_sobel_features():
    rgb = torch.rand(1, 3, 32, 40)
    coarse = torch.rand(1, 1, 32, 40)
    model = build_refine("original")
    model(rgb, coarse)["final"].mean().backward()
