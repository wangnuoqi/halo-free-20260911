"""Resume the local SwinIR protocol with three NCCL DDP workers."""
import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, Subset

from train_paired import PairedJpegDataset, SwinIR, full_loader


class GlobalBatchSampler(Sampler):
    """Give all ranks the same step count, including a partial final batch."""

    def __init__(self, dataset_size, batch_size, rank, world_size, seed):
        if batch_size < world_size:
            raise ValueError('Global batch must be at least the number of ranks')
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        for start in range(0, self.dataset_size, self.batch_size):
            batch = indices[start:start + self.batch_size]
            local_size = (len(batch) + self.world_size - 1) // self.world_size
            padded = batch + [batch[-1]] * (local_size * self.world_size - len(batch))
            begin = self.rank * local_size
            yield padded[begin:begin + local_size]

    def real_count(self, step):
        total = min(self.batch_size, self.dataset_size - step * self.batch_size)
        local_size = (total + self.world_size - 1) // self.world_size
        return max(0, min(local_size, total - self.rank * local_size)), total


def parse_args():
    root = Path(__file__).resolve().parents[3]
    output = Path(__file__).resolve().parent / 'checkpoints/q30_protocol160'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=root / 'openclipart_5000/jpeg_pairs_512')
    parser.add_argument('--list-dir', type=Path, default=root / 'openclipart_5000')
    parser.add_argument('--output-dir', type=Path, default=output)
    parser.add_argument('--resume', type=Path, default=output / 'last.pth')
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--quality', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr-milestones', type=int, nargs='+', default=[40, 80, 120])
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--workers', type=int, default=16, help='Total workers across ranks')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke-steps', type=int, default=0,
                        help='Run this many training steps without validation or checkpoint writes.')
    parser.add_argument('--validate-only', action='store_true',
                        help='Validate the resume checkpoint across all ranks without training or writes.')
    args = parser.parse_args()
    if args.batch_size < int(os.environ.get('WORLD_SIZE', 3)):
        parser.error('Batch size must be at least world size')
    if args.val_every < 1 or args.workers < 0:
        parser.error('val-every must be positive and workers nonnegative')
    if args.smoke_steps < 0:
        parser.error('smoke-steps must be nonnegative')
    return args


def make_model(patch_size):
    window = 7
    model_size = ((patch_size + window - 1) // window) * window
    return SwinIR(img_size=model_size, patch_size=1, in_chans=3, embed_dim=180,
                  depths=[6] * 6, num_heads=[6] * 6, window_size=window,
                  mlp_ratio=2, img_range=255., upscale=1, upsampler='',
                  resi_connection='1conv')


def main():
    args = parse_args()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', init_method='env://')
    try:
        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        previous = checkpoint.get('config', {})
        for key in ('quality', 'patch_size', 'batch_size', 'lr', 'lr_milestones'):
            if previous.get(key) != getattr(args, key):
                raise ValueError(f'Resume {key} mismatch: {previous.get(key)} != {getattr(args, key)}')
        if 'optimizer' not in checkpoint or 'scheduler' not in checkpoint:
            raise ValueError('Resume checkpoint must contain optimizer and scheduler state')
        start_epoch = int(checkpoint['epoch'])
        if start_epoch >= args.epochs:
            raise ValueError(f'Checkpoint has already reached epoch {start_epoch}')
        model = make_model(args.patch_size)
        model.load_state_dict(checkpoint['model'], strict=True)
        model.to(device)
        ddp = DDP(model, device_ids=[local_rank], output_device=local_rank,
                  broadcast_buffers=False)
        optimizer = torch.optim.Adam(ddp.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_milestones, gamma=0.5)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best = float(checkpoint.get('best_val_mse', float('inf')))

        train_set = PairedJpegDataset(args.data_root, args.list_dir, args.quality,
                                      'train', 3, args.patch_size)
        sampler = GlobalBatchSampler(len(train_set), args.batch_size, rank,
                                     world_size, args.seed)
        base_workers, extra = divmod(args.workers, world_size)
        worker_count = base_workers + int(rank < extra)
        train_loader = DataLoader(train_set, batch_sampler=sampler,
                                  num_workers=worker_count, pin_memory=True,
                                  persistent_workers=worker_count > 0)
        val_set = full_loader(args.data_root, args.list_dir, args.quality, 'val', 3)
        val_loader = DataLoader(Subset(val_set, range(rank, len(val_set), world_size)),
                                batch_size=1, num_workers=0)
        history_path = args.output_dir / 'history.json'
        history = json.loads(history_path.read_text()) if rank == 0 and history_path.exists() else []
        if rank == 0:
            if history and history[-1]['epoch'] != start_epoch:
                raise ValueError('History and checkpoint epoch do not match')
            print(f'[resume] epoch={start_epoch} world_size={world_size} '
                  f'global_batch={args.batch_size} steps_per_epoch={len(sampler)} '
                  f'workers={args.workers}', flush=True)
        dist.barrier()

        def validate():
            model.eval()
            local_mse = 0.0
            with torch.inference_mode():
                for index, (low, high) in enumerate(val_loader, 1):
                    pred = model(low.to(device))
                    local_mse += nn.functional.mse_loss(pred, high.to(device)).item()
                    if rank == 0 and index % 25 == 0:
                        print(f'[val] rank 0 image={index}/{len(val_loader)}', flush=True)
            totals = torch.tensor([local_mse, len(val_loader)],
                                  dtype=torch.float64, device=device)
            dist.all_reduce(totals)
            return (totals[0] / totals[1]).item()

        if args.validate_only:
            mse = validate()
            if rank == 0:
                print(f'[validate-only] checkpoint_epoch={start_epoch} '
                      f'images={len(val_set)} mse={mse:.10f}; no checkpoint written', flush=True)
            return

        for epoch in range(start_epoch + 1, args.epochs + 1):
            sampler.epoch = epoch - 1
            ddp.train()
            loss_sum = 0.0
            sample_count = 0
            started = time.perf_counter()
            for step, (low, high) in enumerate(train_loader):
                real_count, global_count = sampler.real_count(step)
                low = low.to(device, non_blocking=True)
                high = high.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                pred = ddp(low)
                per_sample = (pred - high).abs().flatten(1).mean(1)
                local_sum = per_sample[:real_count].sum()
                # DDP averages gradients over ranks; restore the global sample mean.
                (local_sum * world_size / global_count).backward()
                optimizer.step()
                loss_sum += local_sum.detach().item()
                sample_count += real_count
                if rank == 0 and (step == 0 or (step + 1) % 10 == 0 or step + 1 == len(sampler)):
                    print(f'[train] epoch={epoch}/{args.epochs} step={step + 1}/{len(sampler)} '
                          f'elapsed={time.perf_counter() - started:.1f}s', flush=True)
                if args.smoke_steps and step + 1 >= args.smoke_steps:
                    break
            stats = torch.tensor([loss_sum, sample_count], dtype=torch.float64, device=device)
            dist.all_reduce(stats)
            train_l1 = (stats[0] / stats[1]).item()
            if args.smoke_steps:
                if rank == 0:
                    print(f'[smoke] {args.smoke_steps} steps passed, train_l1={train_l1:.6f}; no checkpoint written', flush=True)
                return
            val_mse = None
            improved = False
            if epoch % args.val_every == 0 or epoch == args.epochs:
                val_mse = validate()
                if rank == 0:
                    improved = val_mse < best
                    best = min(best, val_mse)
            if rank == 0:
                row = {'epoch': epoch, 'train_l1': train_l1,
                       'val_mse': val_mse, 'lr': optimizer.param_groups[0]['lr']}
                print(row, flush=True)
            scheduler.step()
            if rank == 0:
                state = {'epoch': epoch, 'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(), 'best_val_mse': best,
                         'config': dict(previous, resume=str(args.resume),
                                        distributed_backend='nccl', world_size=world_size)}
                torch.save(state, args.output_dir / 'last.pth')
                if improved:
                    torch.save(state, args.output_dir / 'best.pth')
                history.append(row)
                history_path.write_text(json.dumps(history, indent=2))
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
