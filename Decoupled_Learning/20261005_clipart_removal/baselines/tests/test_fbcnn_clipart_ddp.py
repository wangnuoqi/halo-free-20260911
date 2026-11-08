import torch

from train_fbcnn_clipart_ddp import (
    DistributedGlobalBatchSampler,
    scale_local_loss,
)


def test_three_rank_sampler_preserves_global_batch_and_epoch_length():
    samplers = [
        DistributedGlobalBatchSampler(
            dataset_size=3722,
            global_batch_size=8,
            rank=rank,
            world_size=3,
            seed=42,
        )
        for rank in range(3)
    ]
    rank_batches = [list(sampler) for sampler in samplers]

    assert [len(batches) for batches in rank_batches] == [465, 465, 465]
    assert [len(batches[0]) for batches in rank_batches] == [2, 3, 3]
    for step_batches in zip(*rank_batches):
        global_batch = [index for local_batch in step_batches for index in local_batch]
        assert len(global_batch) == 8
        assert len(set(global_batch)) == 8

    sampled = {index for batches in rank_batches for batch in batches for index in batch}
    assert len(sampled) == 3720


def test_ddp_local_loss_scaling_matches_global_mean_gradient():
    values = torch.arange(1, 9, dtype=torch.float32)
    targets = values * 0.25
    parameter = torch.tensor(0.7, requires_grad=True)
    global_loss = ((parameter * values - targets) ** 2).mean()
    expected_gradient = torch.autograd.grad(global_loss, parameter)[0]

    gradients = []
    offset = 0
    for local_size in (2, 3, 3):
        local_parameter = torch.tensor(0.7, requires_grad=True)
        local_values = values[offset : offset + local_size]
        local_targets = targets[offset : offset + local_size]
        local_loss = ((local_parameter * local_values - local_targets) ** 2).mean()
        scaled_loss = scale_local_loss(
            local_loss,
            local_batch_size=local_size,
            global_batch_size=8,
            world_size=3,
        )
        gradients.append(torch.autograd.grad(scaled_loss, local_parameter)[0])
        offset += local_size

    ddp_mean_gradient = torch.stack(gradients).mean()
    torch.testing.assert_close(ddp_mean_gradient, expected_gradient)
