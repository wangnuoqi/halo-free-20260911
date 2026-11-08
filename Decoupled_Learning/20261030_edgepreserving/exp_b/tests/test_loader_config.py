from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from exp_b.src.train_exp_b import dataloader_runtime_options, make_protocol


def test_protocol_records_requested_pinned_memory():
    args = SimpleNamespace(
        phase="base",
        data_root="/tmp/data",
        crop_size=128,
        batch_size_per_rank=16,
        precision="bf16",
        lr=3e-4,
        min_lr=3e-6,
        lr_schedule="cosine",
        weight_decay=1e-4,
        seed=42,
        val_count=200,
        base_gradient_weight=0.1,
        epochs=200,
        num_workers=4,
        pin_memory=True,
        base_checkpoint=None,
        refine_checkpoint=None,
    )

    protocol = make_protocol(args, world_size=4)

    assert protocol["pin_memory"] is True


def test_pinned_memory_option_reaches_dataloader():
    loader = DataLoader(
        TensorDataset(torch.zeros(1)),
        batch_size=1,
        **dataloader_runtime_options(num_workers=0, pin_memory=True),
    )

    assert loader.pin_memory is True
    assert loader.persistent_workers is False
