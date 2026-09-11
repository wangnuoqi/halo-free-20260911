import torch

from train_exp6 import refinement_inputs


def test_refinement_inputs_center_both_observations():
    noisy = torch.tensor([[[[0.8]]]])
    coarse = torch.tensor([[[[0.6]]]])

    normalized_noisy, normalized_coarse = refinement_inputs(noisy, coarse)

    assert torch.allclose(normalized_noisy, torch.tensor([[[[0.3]]]]))
    assert torch.allclose(normalized_coarse, torch.tensor([[[[0.1]]]]))
