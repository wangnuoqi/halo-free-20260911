"""Launch the self-contained six-arm full-image Exp7 protocol."""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def run(command):
    print('[exp7]', ' '.join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def prepare_output(output):
    """Return whether an output should run, removing only incomplete attempts."""
    if not output.exists():
        return True
    if (output / 'training_complete.json').is_file():
        print(f'[exp7] completed; skipping {output}', flush=True)
        return False
    print(f'[exp7] removing incomplete output {output}', flush=True)
    shutil.rmtree(output)
    return True


def stage_initializers(models_dir):
    sources = {
        'dncnn': ROOT / 'third_party/DnCNN/TrainingCodes/dncnn_pytorch/models/DnCNN_sigma25/model.pth',
        'ffdnet': ROOT / 'third_party/FFDNet/refine_bridge/pytorch_official/models/ffdnet_gray.pth',
    }
    staged = models_dir / 'init'; staged.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        destination = staged / f'{name}_sigma25_source.pth'
        if not destination.exists(): shutil.copy2(source, destination)
    return {name: staged / f'{name}_sigma25_source.pth' for name in sources}


def base_specs(models_dir, initializers):
    return {
        'convguided': (models_dir / 'convguided_base_full' / 'convguided_best.pth', None),
        'dncnn': (models_dir / 'dncnn_base_full' / 'dncnn_best.pth', initializers['dncnn']),
        'ffdnet': (models_dir / 'ffdnet_base_full' / 'ffdnet_best.pth', initializers['ffdnet']),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root', default=r'D:\nuoqi\halo-free\data_bsds')
    parser.add_argument('--run-root', default=str(ROOT / 'exp7' / 'formal_v2'))
    parser.add_argument('--device', default='cuda:0'); parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--full-batch-size', type=int, default=8)
    parser.add_argument('--only', choices=['all', 'init', 'base', 'joint-rf', 'frozen-rf', 'convguided-base', 'dncnn-base', 'ffdnet-base', 'convguided-joint', 'convguided-frozen', 'dncnn-joint', 'dncnn-frozen', 'ffdnet-joint', 'ffdnet-frozen'], default='all')
    args=parser.parse_args(); root=Path(args.run_root); models=root/'models'; runs=root/'runs'
    initializers=stage_initializers(models)
    if args.only == 'init': return
    data=Path(args.data_root)
    bases=base_specs(models, initializers)
    for denoiser, (base, source) in bases.items():
        if args.only not in {'all', 'base', f'{denoiser}-base'} or not prepare_output(base.parent): continue
        command=[sys.executable, '-u', str(ROOT/'train_convguided_base.py'), '--denoiser', denoiser, '--train-root', str(data/'BSDS 400'), '--test-root', str(data/'BSD68'), '--out', str(base.parent), '--protocol', 'full', '--epochs', '160', '--batch-size', str(args.full_batch_size), '--num-workers', str(args.num_workers), '--device', args.device]
        if source is not None: command += ['--checkpoint', str(source)]
        run(command)
    if args.only == 'base': return
    for mode, rf in (('joint', 'exp7c'), ('frozen', 'exp6d')):
        for denoiser, (checkpoint, _) in bases.items():
            output=runs/f'exp7{"A" if mode == "joint" else "B"}_{denoiser}_{mode}_{rf}_full'
            task=f'{denoiser}-{mode}'
            if args.only not in {'all', task, f'{mode}-rf'} or not prepare_output(output): continue
            run([sys.executable, '-u', str(Path(__file__).with_name('train_exp7.py')), '--denoiser', denoiser, '--mode', mode, '--protocol', 'full', '--rf', rf, '--train-root', str(data/'BSDS 400'), '--test-root', str(data/'BSD68'), '--checkpoint', str(checkpoint), '--out', str(output), '--epochs', '160', '--batch-size', str(args.full_batch_size), '--num-workers', str(args.num_workers), '--device', args.device, '--denoiser-lr', '1e-4', '--refine-lr', '3e-3'])


if __name__ == '__main__': main()
