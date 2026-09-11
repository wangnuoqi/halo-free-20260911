"""Four single-GPU worker scheduler for the formal Exp7/Exp8 matrix."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent


def command(experiment, only, data_root, run_root, workers):
    return [sys.executable, '-u', str(ROOT / experiment / f'run_{experiment}.py'),
            '--data-root', str(data_root), '--run-root', str(run_root), '--num-workers', str(workers),
            '--device', 'cuda:0', '--only', only]


def jobs(experiments):
    matrix = []
    if 'exp7' in experiments:
        matrix += [('exp7-convguided-base', 'exp7', 'convguided-base', ()),
                   ('exp7-dncnn-base', 'exp7', 'dncnn-base', ()),
                   ('exp7-ffdnet-base', 'exp7', 'ffdnet-base', ()),
                   ('exp7-dncnn-joint', 'exp7', 'dncnn-joint', ('exp7-dncnn-base',)),
                   ('exp7-dncnn-frozen', 'exp7', 'dncnn-frozen', ('exp7-dncnn-base',)),
                   ('exp7-ffdnet-joint', 'exp7', 'ffdnet-joint', ('exp7-ffdnet-base',)),
                   ('exp7-ffdnet-frozen', 'exp7', 'ffdnet-frozen', ('exp7-ffdnet-base',)),
                   ('exp7-convguided-joint', 'exp7', 'convguided-joint', ('exp7-convguided-base',)),
                   ('exp7-convguided-frozen', 'exp7', 'convguided-frozen', ('exp7-convguided-base',))]
    if 'exp8' in experiments:
        matrix += [('exp8-base', 'exp8', 'base', ()),
                   ('exp8-dncnn-joint', 'exp8', 'dncnn-joint', ()),
                   ('exp8-dncnn-frozen', 'exp8', 'dncnn-frozen', ()),
                   ('exp8-ffdnet-joint', 'exp8', 'ffdnet-joint', ()),
                   ('exp8-ffdnet-frozen', 'exp8', 'ffdnet-frozen', ()),
                   ('exp8-convguided-joint', 'exp8', 'convguided-joint', ('exp8-base',)),
                   ('exp8-convguided-frozen', 'exp8', 'convguided-frozen', ('exp8-base',))]
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=r'D:\nuoqi\halo-free\data_bsds')
    parser.add_argument('--exp7-root', default=str(ROOT / 'exp7' / 'formal_v2'))
    parser.add_argument('--exp8-root', default=str(ROOT / 'exp8' / 'formal_v2'))
    parser.add_argument('--experiments', nargs='+', choices=['exp7', 'exp8'], default=['exp7', 'exp8'])
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()
    roots = {'exp7': Path(args.exp7_root), 'exp8': Path(args.exp8_root)}
    gpu_slots = [gpu.strip() for gpu in args.gpus.split(',') if gpu.strip()]
    if not gpu_slots: raise ValueError('at least one GPU is required')

    # Stage immutable DnCNN/FFDNet inputs before workers race on the same directory.
    for experiment in args.experiments:
        subprocess.run(command(experiment, 'init', args.data_root, roots[experiment], args.num_workers), check=True)

    pending = {job[0]: job for job in jobs(args.experiments)}
    completed, running = set(), {}
    while pending or running:
        for gpu, (job_id, process, log_file) in list(running.items()):
            status = process.poll()
            if status is None: continue
            log_file.close(); del running[gpu]
            if status:
                raise RuntimeError(f'{job_id} failed on GPU {gpu}; see its log file')
            completed.add(job_id); print(f'[parallel] DONE {job_id} on GPU {gpu}', flush=True)
        for gpu in [slot for slot in gpu_slots if slot not in running]:
            ready_jobs = [job for job in pending.values() if all(dep in completed for dep in job[3])]
            dependent_counts = {job_id: sum(job_id in candidate[3] for candidate in pending.values()) for job_id in pending}
            ready = max(ready_jobs, key=lambda job: (dependent_counts[job[0]], job[0]), default=None)
            if ready is None: break
            job_id, experiment, only, _ = ready; del pending[job_id]
            log_dir = roots[experiment] / 'logs'; log_dir.mkdir(parents=True, exist_ok=True)
            log_file = (log_dir / f'{job_id}.log').open('w', encoding='utf-8')
            environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = gpu; environment['PYTHONUNBUFFERED'] = '1'
            process = subprocess.Popen(command(experiment, only, args.data_root, roots[experiment], args.num_workers), env=environment, stdout=log_file, stderr=subprocess.STDOUT)
            running[gpu] = (job_id, process, log_file); print(f'[parallel] START {job_id} on GPU {gpu}', flush=True)
        if running: time.sleep(5)
        elif pending: raise RuntimeError(f'no runnable jobs; unresolved dependencies: {list(pending)}')


if __name__ == '__main__': main()
