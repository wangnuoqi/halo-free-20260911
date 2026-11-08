"""Windows four-GPU launch and batch semantics."""

import pytest

from baseline.train_nafnet_wls_wintrain import EpochRandomSampler, parse_args, resolve_batch_sizes
from baseline.train_nafnet_wls import build_run_protocol


def test_default_wintrain_paths_and_original_recipe():
    args = parse_args([])
    assert str(args.data_root) == r"D:\ResearchCode\S2R-HDR\data\20260818"
    assert str(args.output_dir).endswith(r"baseline\runs\nafnet_wls_patch256_4gpu")
    assert args.batch_size_per_gpu == 8
    assert args.micro_batch_size == 8
    assert args.train_crop_size == 256
    assert args.total_iter == 200000
    assert args.precision == "bf16"


def test_four_gpu_batch_and_microbatch_preserve_per_card_eight():
    assert resolve_batch_sizes(4, 8, 8) == (32, 32)
    assert resolve_batch_sizes(4, 8, 2) == (32, 8)
    with pytest.raises(ValueError):
        resolve_batch_sizes(4, 8, 9)


def test_checkpoint_protocol_records_four_card_dataparallel():
    args = parse_args([])
    protocol = build_run_protocol(args, ["a", "b"], ["c"], world_size=4, parallelism="DataParallel")
    assert protocol["img_channel"] == 1
    assert protocol["world_size"] == 4
    assert protocol["parallelism"] == "DataParallel"
    assert protocol["effective_batch_size"] == 32


def test_epoch_sampler_replays_the_same_order_after_resume():
    sampler = EpochRandomSampler(list(range(11)), seed=42)
    sampler.set_epoch(3)
    order = list(sampler)
    sampler.set_epoch(4)
    assert list(sampler) != order
    sampler.set_epoch(3)
    assert list(sampler) == order
