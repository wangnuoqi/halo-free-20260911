"""Run the two joint RF loss ablations for one denoiser sequentially."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def prepare_output(output):
    if not output.exists():
        return True
    if (output / 'training_complete.json').is_file():
        print(f'[exp9] completed; skipping {output}', flush=True)
        return False
    print(f'[exp9] removing incomplete output {output}', flush=True)
    shutil.rmtree(output)
    return True


def stage_initializer(models_dir, denoiser):
    sources = {
        'dncnn': ROOT / 'third_party/DnCNN/TrainingCodes/dncnn_pytorch/models/DnCNN_sigma25/model.pth',
        'ffdnet': ROOT / 'third_party/FFDNet/refine_bridge/pytorch_official/models/ffdnet_gray.pth',
    }
    destination = models_dir / 'init' / f'{denoiser}_sigma25_source.pth'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(sources[denoiser], destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--denoiser', choices=['dncnn', 'ffdnet'], required=True)
    parser.add_argument('--data-root', default=r'D:\nuoqi\halo-free\data_bsds')
    parser.add_argument('--run-root', default=str(ROOT / 'exp9' / 'formal_v1'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()

    settings = {
        'dncnn': ('dncnn_official', 128, 50, 1e-3, 0.0),
        'ffdnet': ('ffdnet_official', 64, 500, 1e-4, 5e-4),
    }
    protocol, batch_size, epochs, denoiser_lr, weight_decay = settings[args.denoiser]
    root = Path(args.run_root)
    data = Path(args.data_root)
    checkpoint = stage_initializer(root / 'models', args.denoiser)
    runs = root / 'runs'

    for loss_mode, suffix in (('mse_smooth', 'mse_smooth'), ('mse', 'mse')):
        output = runs / f'exp9_{args.denoiser}_joint_{suffix}_patch'
        if not prepare_output(output):
            continue
        command = [
            sys.executable, '-u', str(ROOT / 'exp9/train_exp9.py'),
            '--denoiser', args.denoiser, '--mode', 'joint', '--protocol', protocol,
            '--rf', 'exp7c', '--train-root', str(data / 'BSDS 400'),
            '--test-root', str(data / 'BSD68'), '--checkpoint', str(checkpoint),
            '--out', str(output), '--epochs', str(epochs), '--batch-size', str(batch_size),
            '--num-workers', str(args.num_workers), '--device', args.device,
            '--denoiser-lr', str(denoiser_lr), '--denoiser-weight-decay', str(weight_decay),
            '--refine-lr', '3e-3', '--coarse-loss', loss_mode,
        ]
        print('[exp9]', ' '.join(map(str, command)), flush=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
