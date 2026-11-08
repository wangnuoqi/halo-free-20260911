"""Resume the OpenClipart FBCNN run with local NCCL DistributedDataParallel."""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'FBCNN'))
from models.network_fbcnn import FBCNN
from models.select_network import init_weights


class Pairs(Dataset):
    def __init__(self, data, lists, split, quality, patch_size=128):
        self.train = split == 'train'
        self.patch = patch_size
        self.q = (100 - quality) / 100.0
        stems = [
            line.strip()
            for line in (lists / f'{split}_list.txt').read_text().splitlines()
            if line.strip()
        ]
        if not stems or len(stems) != len(set(stems)):
            raise ValueError(f'Empty or duplicated {split} split list')
        self.pairs = [
            (data / f'jpeg_q{quality:02d}/{stem}.jpg', data / f'clean_png/{stem}.png')
            for stem in stems
        ]
        for pair in self.pairs:
            for path in pair:
                if not path.is_file():
                    raise FileNotFoundError(path)
        print(f'{split}: {len(self.pairs)} pairs, Q{quality}', flush=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        arrays = []
        for path in self.pairs[index]:
            with Image.open(path) as image:
                arrays.append(np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0)
        if arrays[0].shape != arrays[1].shape:
            raise ValueError(f'Unaligned pair {self.pairs[index]}')
        if self.train:
            height, width = arrays[0].shape[:2]
            if min(height, width) < self.patch:
                raise ValueError(f'Image smaller than patch: {self.pairs[index]}')
            y = random.randint(0, height - self.patch)
            x = random.randint(0, width - self.patch)
            arrays = [array[y:y + self.patch, x:x + self.patch] for array in arrays]
        tensors = [
            torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
            for array in arrays
        ]
        return tensors[0], tensors[1], torch.tensor([self.q], dtype=torch.float32)


class DistributedGlobalBatchSampler(Sampler):
    """Shard one shuffled global batch across DDP ranks without dropping samples."""

    def __init__(self, dataset_size, global_batch_size, rank, world_size, seed=42):
        if world_size < 2 or not 0 <= rank < world_size:
            raise ValueError('DDP sampler requires world_size >= 2 and a valid rank')
        if global_batch_size < world_size:
            raise ValueError('Global batch size must be at least the world size')
        self.dataset_size = dataset_size
        self.global_batch_size = global_batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        base, remainder = divmod(global_batch_size, world_size)
        self.local_batch_sizes = [base] * world_size
        for extra_rank in range(world_size - remainder, world_size):
            self.local_batch_sizes[extra_rank] += 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.dataset_size // self.global_batch_size

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        usable = len(self) * self.global_batch_size
        indices = indices[:usable]
        start = sum(self.local_batch_sizes[:self.rank])
        stop = start + self.local_batch_sizes[self.rank]
        for batch_start in range(0, usable, self.global_batch_size):
            global_batch = indices[batch_start:batch_start + self.global_batch_size]
            yield global_batch[start:stop]


def scale_local_loss(local_mean_loss, local_batch_size, global_batch_size, world_size):
    """Undo DDP's rank averaging while preserving a global sample mean."""
    return local_mean_loss * (local_batch_size * world_size / global_batch_size)


def parse_args():
    project = ROOT.parent
    default_output = ROOT / 'ckpt/fbcnn_rgb_q30_protocol160'
    parser = argparse.ArgumentParser(description='Resume FBCNN with three NCCL DDP ranks')
    parser.add_argument('--data-root', type=Path, default=project / 'openclipart_5000/jpeg_pairs_512')
    parser.add_argument('--list-dir', type=Path, default=project / 'openclipart_5000')
    parser.add_argument('--output-dir', type=Path, default=default_output)
    parser.add_argument('--resume', type=Path, default=default_output / 'fbcnn_last.pth')
    parser.add_argument('--quality', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--batch-size', type=int, default=8, help='global batch across all ranks')
    parser.add_argument('--num-workers', type=int, default=16, help='total workers across all ranks')
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--world-size', type=int, default=3)
    parser.add_argument('--master-port', type=str, default='29631')
    args = parser.parse_args()
    if args.val_every < 1:
        parser.error('--val-every must be >= 1')
    if args.batch_size < args.world_size:
        parser.error('--batch-size must be at least --world-size')
    return args


def launch_workers(args):
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
        raise RuntimeError(
            f'Requested {args.world_size} GPUs but found {torch.cuda.device_count()}'
        )
    base_env = os.environ.copy()
    base_env['MASTER_ADDR'] = '127.0.0.1'
    base_env['MASTER_PORT'] = args.master_port
    base_env['WORLD_SIZE'] = str(args.world_size)
    base_env['USE_LIBUV'] = '0'
    command = [sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:]]
    processes = []
    try:
        for rank in range(args.world_size):
            worker_env = base_env.copy()
            worker_env['RANK'] = str(rank)
            worker_env['LOCAL_RANK'] = str(rank)
            print(f'Launching NCCL worker {rank}/{args.world_size}', flush=True)
            processes.append(subprocess.Popen(command, env=worker_env))

        while True:
            exit_codes = [process.poll() for process in processes]
            failed = next((code for code in exit_codes if code not in (None, 0)), None)
            if failed is not None:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    process.wait()
                return failed
            if all(code is not None for code in exit_codes):
                return 0
            time.sleep(0.2)
    except KeyboardInterrupt:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        return 130


def make_model():
    return FBCNN(
        in_nc=3,
        out_nc=3,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode='BR',
        downsample_mode='strideconv',
        upsample_mode='convtranspose',
    )


def train_worker(args):
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    try:
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed(args.seed + rank)
        torch.backends.cudnn.benchmark = True

        if not args.resume.is_file():
            raise FileNotFoundError(args.resume)
        saved = torch.load(args.resume, map_location='cpu')
        saved_config = saved['config']
        for key, value in (
            ('quality', args.quality),
            ('patch_size', args.patch_size),
            ('batch_size', args.batch_size),
        ):
            if saved_config.get(key) != value:
                raise ValueError(
                    f'Resume {key} mismatch: checkpoint={saved_config.get(key)}, current={value}'
                )

        raw_model = make_model()
        raw_model.load_state_dict(saved['model_state_dict'], strict=True)
        raw_model.to(device)
        model = DistributedDataParallel(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_set = Pairs(args.data_root, args.list_dir, 'train', args.quality, args.patch_size)
        val_set = Pairs(args.data_root, args.list_dir, 'val', args.quality, args.patch_size) if rank == 0 else None
        steps_per_epoch = len(train_set) // args.batch_size
        if steps_per_epoch == 0:
            raise ValueError('No complete training batches')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[40 * steps_per_epoch, 80 * steps_per_epoch, 120 * steps_per_epoch],
            gamma=0.5,
        )
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        start_epoch = int(saved['epoch'])
        global_step = int(saved['global_step'])
        best = float(saved['best_val_mse'])
        if start_epoch >= args.epochs:
            raise ValueError(f'Checkpoint already reached epoch {start_epoch}')

        sampler = DistributedGlobalBatchSampler(
            len(train_set), args.batch_size, rank, world_size, args.seed
        )
        base_workers, extra_workers = divmod(args.num_workers, world_size)
        num_workers = base_workers + int(rank < extra_workers)
        train_loader = DataLoader(
            train_set,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            pin_memory=True,
        ) if rank == 0 else None

        args.output_dir.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            config = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            }
            config['distributed_backend'] = 'nccl'
            config['resume_epoch'] = start_epoch
            (args.output_dir / 'config_used.json').write_text(
                json.dumps(config, indent=2), encoding='utf-8'
            )
            print(
                f'Resuming epoch {start_epoch}, iter {global_step}; '
                f'world_size={world_size}, global batch={args.batch_size}, '
                f'local batches={sampler.local_batch_sizes}, workers={args.num_workers}',
                flush=True,
            )
        dist.barrier()

        for epoch in range(start_epoch, args.epochs):
            sampler.set_epoch(epoch)
            model.train()
            epoch_loss_sum = 0.0
            epoch_sample_count = 0
            bar = tqdm(
                train_loader,
                desc=f'FBCNN DDP train {epoch + 1}/{args.epochs}',
                disable=rank != 0,
            )
            for x, gt, q in bar:
                x = x.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                q = q.to(device, non_blocking=True)
                for buffer in raw_model.buffers():
                    dist.broadcast(buffer, src=0)
                optimizer.zero_grad(set_to_none=True)
                local_batch = x.shape[0]
                pred, q_est = model(x)
                image_loss = F.l1_loss(pred, gt)
                q_loss = F.l1_loss(q_est, q)
                loss = image_loss + 0.001 * q_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                scaled_loss = scale_local_loss(
                    loss, local_batch, args.batch_size, world_size
                )
                scaled_loss.backward()
                optimizer.step()
                scheduler.step()
                global_step += 1
                epoch_loss_sum += loss.item() * local_batch
                epoch_sample_count += local_batch
                if rank == 0:
                    bar.set_postfix(
                        loss=f'{loss.item():.5f}',
                        q_loss=f'{q_loss.item():.4f}',
                        lr=f'{optimizer.param_groups[0]["lr"]:.1e}',
                    )
            bar.close()

            stats = torch.tensor(
                [epoch_loss_sum, float(epoch_sample_count)],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            train_loss = (stats[0] / stats[1]).item()
            should_validate = (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs
            mse = None
            improved = False
            psnrs = []
            q_error = 0.0

            if should_validate:
                dist.barrier()
                if rank == 0:
                    raw_model.eval()
                    sse = 0.0
                    elements = 0
                    with torch.inference_mode():
                        for x, gt, q in tqdm(val_loader, desc='FBCNN DDP val full images'):
                            x = x.to(device, non_blocking=True)
                            gt = gt.to(device, non_blocking=True)
                            q = q.to(device, non_blocking=True)
                            pred, q_est = raw_model(x)
                            diff = (pred.clamp(0, 1) - gt).square()
                            image_mse = diff.mean().item()
                            sse += diff.sum().item()
                            elements += gt.numel()
                            psnrs.append(-10 * math.log10(max(image_mse, 1e-12)))
                            q_error += F.l1_loss(q_est, q).item()
                    mse = sse / elements
                    improved = mse < best
                    best = min(best, mse)
                dist.barrier()

            if rank == 0:
                state = {
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'best_val_mse': best,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config,
                }
                torch.save(state, args.output_dir / 'fbcnn_last.pth')
                if improved:
                    torch.save(state, args.output_dir / 'fbcnn_best.pth')
                record = {
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'train_loss': train_loss,
                    'val_mse': mse,
                    'val_psnr_global': -10 * math.log10(max(mse, 1e-12)) if mse is not None else None,
                    'val_psnr_mean': sum(psnrs) / len(psnrs) if psnrs else None,
                    'val_q_mae': q_error / len(psnrs) if psnrs else None,
                }
                with (args.output_dir / 'history.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record) + '\n')
                print(record, flush=True)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    args = parse_args()
    if 'RANK' in os.environ:
        train_worker(args)
        return
    raise SystemExit(launch_workers(args))


if __name__ == '__main__':
    main()
