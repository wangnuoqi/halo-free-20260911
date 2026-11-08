from types import SimpleNamespace
import pytest
import torch
from expe import refine_joint as chain
from exp_b.src.train_exp_b import set_trainable_phase, forward_phase, delta_cascade_loss

torch.set_num_threads(1)


def test_base_adapter_and_phase_gradients():
    model = chain.FullExpE('viv_gf_e1')
    lr, hr = torch.rand(1, 1, 16, 16), torch.rand(1, 1, 32, 32)
    model.eval()
    with torch.no_grad():
        expected = model.base.network(hr, lr)
        actual = model.base(lr - .5, hr - .5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    selected = set_trainable_phase(model, 'refine')
    outputs = forward_phase('refine', model, selected, lr, hr)
    delta_cascade_loss(outputs, torch.rand_like(hr))[0].backward()
    assert all(p.grad is None for p in model.base.parameters())
    assert any(p.grad is not None for p in model.refine.parameters())
    model.zero_grad(set_to_none=True)
    set_trainable_phase(model, 'joint')
    outputs = forward_phase('joint', model, model, lr, hr)
    delta_cascade_loss(outputs, torch.rand_like(hr))[0].backward()
    assert any(p.grad is not None for p in model.base.parameters())


def test_load_e1_then_joint_and_reject_wrong_base(tmp_path):
    model = chain.FullExpE('viv_gf_e1')
    base = tmp_path / 'best.pth'
    torch.save({'model': model.base.network.state_dict(), 'protocol': {'variant': 'viv_gf_e1'}}, base)
    args = SimpleNamespace(phase='refine', base_checkpoint=str(base), refine_checkpoint=None, expe_variant='viv_gf_e1')
    chain.initialize_phase(model, args, {})
    refined = tmp_path / 'refine_best.pth'
    protocol = {'architecture': 'expe_viv_gf_e1_three_stage_refine', 'base_checkpoint_sha256': chain.file_sha256(base)}
    torch.save({'model_state': model.state_dict(), 'phase': 'refine', 'protocol': protocol}, refined)
    args.phase, args.refine_checkpoint = 'joint', str(refined)
    restored = chain.FullExpE('viv_gf_e1')
    chain.initialize_phase(restored, args, protocol)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    torch.save({'model': restored.base.network.state_dict(), 'protocol': {'variant': 'wrong'}}, base)
    with pytest.raises(ValueError, match='variant'):
        chain.initialize_phase(restored, args, protocol)


def test_joint_ddp_two_iterations(tmp_path):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    dist.init_process_group('gloo', init_method=(tmp_path / 'rendezvous').as_uri(), rank=0, world_size=1)
    try:
        model = chain.FullExpE('viv_gf_e1')
        wrapped = DistributedDataParallel(model, find_unused_parameters=True)
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            lr, hr = torch.rand(1, 1, 16, 16), torch.rand(1, 1, 32, 32)
            outputs = forward_phase('joint', model, wrapped, lr, hr)
            delta_cascade_loss(outputs, torch.rand_like(hr))[0].backward()
        assert model.base.network.lr.reduce_chan_level2.weight.grad is None
    finally:
        dist.destroy_process_group()


def test_cpu_refine_then_joint_end_to_end(tmp_path, monkeypatch):
    import cv2
    import hashlib
    import json
    import sys
    import numpy as np
    from exp_b.src import train_exp_b as engine
    from exp_b.src.data_exp_b import DATA_DIRS, deterministic_train_val_indices
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    data = tmp_path / 'data'
    ids = ['a-log', 'b-log', 'c-log']
    for folder in DATA_DIRS['train']:
        (data / folder).mkdir(parents=True)
        for name in ids:
            edge = 16 if 'downsample' in folder else 32
            cv2.imwrite(str(data / folder / (name + '.tif')), np.random.default_rng(42).random((edge, edge), dtype=np.float32))
    ti, vi = deterministic_train_val_indices(ids, 1, 42)
    protocol = {'variant': 'viv_gf_e1', 'data_root': str(data),
                'train_ids_sha256': hashlib.sha256('\n'.join(ids[i][:-4] for i in ti).encode()).hexdigest(),
                'validation_ids_sha256': hashlib.sha256('\n'.join(ids[i][:-4] for i in vi).encode()).hexdigest()}
    base = tmp_path / 'base.pth'
    torch.save({'model': chain.FullExpE('viv_gf_e1').base.network.state_dict(), 'protocol': protocol}, base)
    for phase in ['refine', 'joint']:
        argv = ['train', '--phase', phase, '--expe-variant', 'viv_gf_e1', '--data-root', str(data),
                '--base-checkpoint', str(base), '--output-dir', str(tmp_path / phase), '--crop-size', '32',
                '--batch-size-per-rank', '1', '--epochs', '1', '--num-workers', '0', '--precision', 'fp32',
                '--val-count', '1', '--parallelism', 'single', '--device', 'cpu', '--full-image-validation']
        if phase == 'joint':
            argv += ['--refine-checkpoint', str(tmp_path / 'refine/refine_best.pth')]
        monkeypatch.setattr(sys, 'argv', argv)
        engine.main()
        latest = torch.load(tmp_path / phase / f'{phase}_latest.pth', weights_only=False)
        best = torch.load(tmp_path / phase / f'{phase}_best.pth', weights_only=False)
        assert latest['best_val_mse'] == best['best_val_mse']
        assert latest['step'] == 2
        assert latest['protocol']['validation_geometry'] == 'full_image'
        assert (tmp_path / phase / 'train_steps.jsonl').is_file()


def test_ddp_unused_detection_is_limited_to_expe_joint():
    from exp_b.src.train_exp_b import needs_unused_parameter_detection
    assert not needs_unused_parameter_detection('refine', 'viv_gf_e1')
    assert needs_unused_parameter_detection('joint', 'viv_gf_e1')
    assert not needs_unused_parameter_detection('joint', None)


def test_refine_ddp_two_iterations_without_unused_detection(tmp_path):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    dist.init_process_group('gloo', init_method=(tmp_path / 'refine_rendezvous').as_uri(), rank=0, world_size=1)
    try:
        model = chain.FullExpE('viv_gf_e1')
        selected = set_trainable_phase(model, 'refine')
        wrapped = DistributedDataParallel(selected, find_unused_parameters=False)
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            lr, hr = torch.rand(1, 1, 16, 16), torch.rand(1, 1, 32, 32)
            outputs = forward_phase('refine', model, wrapped, lr, hr)
            delta_cascade_loss(outputs, torch.rand_like(hr))[0].backward()
            assert all(p.grad is not None for p in selected.parameters())
    finally:
        dist.destroy_process_group()
