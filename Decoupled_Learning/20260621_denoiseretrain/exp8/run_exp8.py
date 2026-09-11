"""Launch the self-contained six-arm patch-training Exp8 protocol."""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).parents[1]


def select_modes(selection, denoiser):
    if selection in {'all', f'{denoiser}-rf'}:
        return ('frozen', 'joint')
    if selection == f'{denoiser}-frozen':
        return ('frozen',)
    if selection == f'{denoiser}-joint':
        return ('joint',)
    return ()


def run(command):
    print('[exp8]', ' '.join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def prepare_output(output):
    """Return whether an output should run, removing only incomplete attempts."""
    if not output.exists():
        return True
    if (output / 'training_complete.json').is_file():
        print(f'[exp8] completed; skipping {output}', flush=True)
        return False
    print(f'[exp8] removing incomplete output {output}', flush=True)
    shutil.rmtree(output)
    return True


def stage_initializers(models_dir):
    sources={'dncnn':ROOT/'third_party/DnCNN/TrainingCodes/dncnn_pytorch/models/DnCNN_sigma25/model.pth', 'ffdnet':ROOT/'third_party/FFDNet/refine_bridge/pytorch_official/models/ffdnet_gray.pth'}
    staged=models_dir/'init'; staged.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        destination=staged/f'{name}_sigma25_source.pth'
        if not destination.exists(): shutil.copy2(source, destination)
    return {name:staged/f'{name}_sigma25_source.pth' for name in sources}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data-root', default=r'D:\nuoqi\halo-free\data_bsds'); parser.add_argument('--run-root', default=str(ROOT/'exp8'/'formal_v2')); parser.add_argument('--device', default='cuda:0'); parser.add_argument('--num-workers', type=int, default=8); parser.add_argument('--only', choices=['all', 'init', 'base', 'convguided-joint', 'convguided-frozen', 'convguided-rf', 'dncnn-joint', 'dncnn-frozen', 'dncnn-rf', 'ffdnet-joint', 'ffdnet-frozen', 'ffdnet-rf'], default='all')
    args=parser.parse_args(); root=Path(args.run_root); models=root/'models'; runs=root/'runs'; data=Path(args.data_root)
    initializers=stage_initializers(models); base=models/'convguided_base_patch128'/'convguided_best.pth'
    if args.only == 'init': return
    if args.only in {'all', 'base'} and prepare_output(base.parent):
        run([sys.executable, '-u', str(ROOT/'train_convguided_base.py'), '--denoiser', 'convguided', '--train-root', str(data/'BSDS 400'), '--test-root', str(data/'BSD68'), '--out', str(base.parent), '--protocol', 'patch', '--epochs', '80', '--batch-size', '32', '--patch-size', '128', '--patches-per-image', '32', '--num-workers', str(args.num_workers), '--device', args.device])
    if args.only == 'base': return
    specs=[('A', 'convguided', base, 'convguided_patch', 32, 80, 1e-4, 0.0), ('B', 'dncnn', initializers['dncnn'], 'dncnn_official', 128, 50, 1e-3, 0.0), ('C', 'ffdnet', initializers['ffdnet'], 'ffdnet_official', 64, 500, 1e-4, 5e-4)]
    for letter, denoiser, checkpoint, protocol, batch_size, epochs, denoiser_lr, weight_decay in specs:
        for mode in select_modes(args.only, denoiser):
            rf = 'exp6d' if mode == 'frozen' else 'exp7c'
            output=runs/f'exp8{letter}_{denoiser}_{mode}_{rf}_patch'
            run_epochs = 160 if mode == 'frozen' else epochs
            if not prepare_output(output):
                continue
            run([sys.executable, '-u', str(ROOT/'exp8/train_exp8.py'), '--denoiser', denoiser, '--mode', mode, '--protocol', protocol, '--rf', rf, '--train-root', str(data/'BSDS 400'), '--test-root', str(data/'BSD68'), '--checkpoint', str(checkpoint), '--out', str(output), '--epochs', str(run_epochs), '--batch-size', str(batch_size), '--num-workers', str(args.num_workers), '--device', args.device, '--denoiser-lr', str(denoiser_lr), '--denoiser-weight-decay', str(weight_decay), '--refine-lr', '3e-3'])


if __name__ == '__main__': main()
