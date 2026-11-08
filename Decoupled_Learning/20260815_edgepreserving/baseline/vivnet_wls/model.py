"""Single-channel VIVNet construction and padding helpers."""

import torch
import torch.nn.functional as F

from .VIVNet_arch import VIVNet


def build_model(device):
    return VIVNet(
        inp_channels=1,
        out_channels=1,
        dim=32,
        num_blocks=[1, 1, 1],
        num_refinement_blocks=4,
        ffn_expansion_factor=3,
        bias=False,
    ).to(device)


def pad_to_multiple(image: torch.Tensor, multiple: int = 4):
    height, width = image.shape[-2:]
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple
    if (pad_height and pad_height >= height) or (pad_width and pad_width >= width):
        raise ValueError("Reflection padding requires each spatial side to exceed its padding")
    padded = F.pad(image, (0, pad_width, 0, pad_height), mode="reflect")
    return padded, (height, width)


def run_padded(model, image: torch.Tensor):
    padded, (height, width) = pad_to_multiple(image)
    return model(padded)[..., :height, :width]
