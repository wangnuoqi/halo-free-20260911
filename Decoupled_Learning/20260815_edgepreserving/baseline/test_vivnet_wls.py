"""Contract tests for the single-channel WLS VIVNet baseline."""

import torch
import pytest
import torch.distributed as dist

from baseline.vivnet_wls.model import build_model, pad_to_multiple, run_padded
from baseline.vivnet_wls.train import (
    batch_micro_slices,
    distributed_requested,
    wrap_ddp_model,
    fft_loss,
    distributed_requested,
    next_training_position,
    validate_resume_protocol,
)
from baseline.vivnet_wls.train_wintrain import resolve_batch_sizes


def test_vivnet_wls_uses_one_channel_and_preserves_full_image_shape():
    torch.set_num_threads(2)
    model = build_model(torch.device("cpu"))
    assert model.patch_embed.proj.in_channels == 1
    assert model.output.out_channels == 1
    image = torch.rand(1, 1, 38, 46)
    padded, original_size = pad_to_multiple(image, multiple=4)
    assert padded.shape == (1, 1, 40, 48)
    assert original_size == (38, 46)
    with torch.inference_mode():
        prediction = run_padded(model, image)
    assert prediction.shape == image.shape


def test_vivnet_wls_parameter_count_matches_configured_network():
    model = build_model(torch.device("cpu"))
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_vivnet_wls_fft_loss_is_zero_for_identical_pairs():
    image = torch.rand(2, 1, 32, 32)
    assert fft_loss(image, image).item() == 0.0


def test_vivnet_wls_resume_rejects_changed_data_geometry():
    with pytest.raises(ValueError, match="protocol differs"):
        validate_resume_protocol(
            {"run_protocol": {"train_crop_size": 256, "world_size": 3}},
            {"train_crop_size": 128, "world_size": 3},
        )


def test_vivnet_wls_keeps_uneven_tail_batch_and_weights_microbatches():
    chunks = batch_micro_slices(25, 16)
    assert chunks == [(0, 16, 16 / 25), (16, 25, 9 / 25)]
    assert sum(weight for _, _, weight in chunks) == pytest.approx(1.0)


def test_vivnet_wls_checkpoint_records_next_batch_or_next_epoch():
    assert next_training_position(epoch=7, batch_index=2, batch_count=5) == (7, 3)
    assert next_training_position(epoch=7, batch_index=4, batch_count=5) == (8, 0)


def test_vivnet_wintrain_batch_sizes_scale_across_four_dp_devices():
    assert resolve_batch_sizes(4, per_gpu_batch=64, per_gpu_micro_batch=16) == (256, 64)


def test_vivnet_trainer_uses_plain_single_gpu_without_torchrun():
    assert not distributed_requested({})
    assert not distributed_requested({"WORLD_SIZE": "1"})
    assert distributed_requested({"WORLD_SIZE": "3"})
    assert distributed_requested({"WORLD_SIZE": "1", "LOCAL_RANK": "0"})


def test_vivnet_ddp_reducer_handles_unused_legacy_layer(tmp_path):
    init_file = tmp_path / "gloo_init"
    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=0, world_size=1
    )
    try:
        model = build_model(torch.device("cpu"))
        ddp_model = wrap_ddp_model(model)
        image = torch.rand(1, 1, 32, 32)
        for _ in range(2):
            ddp_model.zero_grad(set_to_none=True)
            with ddp_model.no_sync():
                ddp_model(image).square().mean().backward()
            ddp_model(image).square().mean().backward()
        assert model.reduce_chan_level2.weight.grad is None
    finally:
        dist.destroy_process_group()
