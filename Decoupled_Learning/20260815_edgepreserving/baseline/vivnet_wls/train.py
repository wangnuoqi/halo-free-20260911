"""Train single-channel VIVNet on paired WLS images, singly or with NCCL DDP."""

import argparse
import csv
from contextlib import nullcontext
from functools import partial
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baseline.wls_data import WLSPairs, paired_patch_collate, partition_ids, collate_pairs
from baseline.vivnet_wls.model import build_model, run_padded


DEFAULT_DATA = Path("/mnt/d/ResearchCode/S2R-HDR/data/20260818")
DEFAULT_OUTPUT = ROOT / "baseline/runs/vivnet_wls_patch256_nccl_3gpu_bs32"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total-iter", type=int, default=200000)
    parser.add_argument("--batch-size-per-gpu", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--train-crop-size", type=int, default=256)
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="CUDA device index for direct single-GPU execution")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--val-freq", type=int, default=20000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def distributed_requested(environ=None):
    environ = os.environ if environ is None else environ
    return int(environ.get("WORLD_SIZE", "1")) > 1 or "LOCAL_RANK" in environ


def wrap_ddp_model(model, local_rank=None):
    # The upstream architecture keeps reduce_chan_level2 even though forward skips it.
    options = {"find_unused_parameters": True}
    if local_rank is not None:
        options.update(device_ids=[local_rank], output_device=local_rank)
    return DistributedDataParallel(model, **options)


def protocol_for(args, world_size, train_ids, val_ids):
    return {
        "architecture": "VIVNet-T-single-channel",
        "task": "single_channel_WLS_l0p500_a1p400",
        "inp_channels": 1,
        "out_channels": 1,
        "source_data_root": str(args.data_root.resolve()),
        "train_pairs": len(train_ids),
        "validation_pairs": len(val_ids),
        "validation_source": "deterministic_holdout_from_train",
        "train_ids_sha256": __import__("hashlib").sha256("\n".join(train_ids).encode()).hexdigest(),
        "validation_ids_sha256": __import__("hashlib").sha256("\n".join(val_ids).encode()).hexdigest(),
        "train_geometry": "random_paired_patch",
        "train_crop_size": args.train_crop_size,
        "validation_geometry": "full_image_even_padded_to_multiple_of_4",
        "channel_order": "grayscale",
        "value_range": "0_1",
        "parallelism": "DDP" if distributed_requested() else "single_gpu",
        "world_size": world_size,
        "batch_size_per_gpu": args.batch_size_per_gpu,
        "effective_batch_size": world_size * args.batch_size_per_gpu,
        "micro_batch_size": args.micro_batch_size,
        "precision": args.precision,
        "total_iter": args.total_iter,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "betas": [args.beta1, args.beta2],
        "scheduler": "CosineAnnealingLR",
        "min_lr": args.min_lr,
        "loss": "L1Loss + 0.1 * FFTLoss",
        "val_freq": args.val_freq,
        "save_freq": args.save_freq,
        "seed": args.seed,
        "val_count": args.val_count,
    }


def validate_resume_protocol(saved, current):
    previous = dict(saved.get("run_protocol", {}))
    requested = dict(current)
    previous.pop("micro_batch_size", None)
    requested.pop("micro_batch_size", None)
    # A run may migrate between single GPU and DDP as long as the global
    # batch stays unchanged. This preserves optimizer/scheduler semantics;
    # changing the effective batch would make a direct resume incomparable.
    previous_effective_batch = previous.get(
        "effective_batch_size",
        previous.get("world_size", 1) * previous.get("batch_size_per_gpu", 1),
    )
    requested_effective_batch = requested.get(
        "effective_batch_size",
        requested.get("world_size", 1) * requested.get("batch_size_per_gpu", 1),
    )
    if previous_effective_batch != requested_effective_batch:
        raise ValueError(
            "Checkpoint effective batch differs; keep effective batch size when migrating parallelism"
        )
    for protocol in (previous, requested):
        protocol.pop("parallelism", None)
        protocol.pop("world_size", None)
        protocol.pop("batch_size_per_gpu", None)
        protocol.pop("effective_batch_size", None)
    if previous != requested:
        raise ValueError("Checkpoint training protocol differs; use matching data, geometry and optimizer settings")


def batch_micro_slices(batch_size, micro_batch_size):
    if batch_size < 1 or micro_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    return [
        (start, min(start + micro_batch_size, batch_size), min(micro_batch_size, batch_size - start) / batch_size)
        for start in range(0, batch_size, micro_batch_size)
    ]


def next_training_position(epoch, batch_index, batch_count):
    if batch_index + 1 >= batch_count:
        return epoch + 1, 0
    return epoch, batch_index + 1


def fft_loss(prediction, target):
    pred_fft = torch.fft.fft2(prediction.float(), dim=(-2, -1))
    target_fft = torch.fft.fft2(target.float(), dim=(-2, -1))
    pred_pair = torch.stack((pred_fft.real, pred_fft.imag), dim=-1)
    target_pair = torch.stack((target_fft.real, target_fft.imag), dim=-1)
    return F.l1_loss(pred_pair, target_pair)


def evaluate(model, loader, device):
    rows = []
    model.eval()
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            prediction = run_padded(model, inputs).float().clamp(0.0, 1.0)
            error = prediction - targets.float()
            mse = error.square().flatten(1).mean(1)
            mae = error.abs().flatten(1).mean(1)
            rows.extend((float(a), float(b), float(-10.0 * torch.log10(a.clamp_min(1e-12))))
                        for a, b in zip(mse.cpu(), mae.cpu()))
    model.train()
    return {
        "images": len(rows),
        "mse": float(np.mean([row[0] for row in rows])),
        "mae": float(np.mean([row[1] for row in rows])),
        "mean_psnr_db": float(np.mean([row[2] for row in rows])),
    }


def save_checkpoint(path, model, optimizer, scheduler, iteration, epoch, next_batch, protocol, best_val_mse):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
    torch.save({
        "model_state": state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "iter": iteration,
        "epoch": epoch,
        "next_batch": next_batch,
        "best_val_mse": best_val_mse,
        "run_protocol": protocol,
    }, path)


def main():
    args = parse_args()
    if args.batch_size_per_gpu % args.micro_batch_size:
        raise ValueError("batch-size-per-gpu must be divisible by micro-batch-size")
    distributed = distributed_requested()
    requested_world_size = int(os.environ.get("WORLD_SIZE", "1")) if distributed else 1
    if distributed and requested_world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = args.gpu_id
    if not 0 <= local_rank < torch.cuda.device_count():
        raise ValueError(f"CUDA device index {local_rank} is outside the visible device range")
    torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    device = torch.device("cuda", local_rank)

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True

    dataset = WLSPairs(args.data_root, "train")
    train_ids, val_ids = partition_ids(dataset.ids, args.val_count, args.seed)
    index_by_id = {identifier: index for index, identifier in enumerate(dataset.ids)}
    train_set = Subset(dataset, [index_by_id[key] for key in train_ids])
    val_set = Subset(dataset, [index_by_id[key] for key in val_ids])
    protocol = protocol_for(args, world_size, train_ids, val_ids)

    sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True,
        drop_last=False, seed=args.seed,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size_per_gpu, sampler=sampler,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
        collate_fn=partial(paired_patch_collate, crop_size=args.train_crop_size),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_pairs)

    model = build_model(device)
    if distributed:
        model = wrap_ddp_model(model, local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_iter, eta_min=args.min_lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.output_dir / "vivnet_wls_latest.pth"
    best_path = args.output_dir / "vivnet_wls_best.pth"
    start_iter = 0
    best_val_mse = float("inf")
    start_epoch = 0
    start_batch = 0
    if args.resume:
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        validate_resume_protocol(checkpoint, protocol)
        checkpoint_model = model.module if distributed else model
        checkpoint_model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_iter = int(checkpoint["iter"])
        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint.get("next_batch", 0))
        best_val_mse = float(checkpoint.get("best_val_mse", best_val_mse))

    if rank == 0:
        (args.output_dir / "config_used.json").write_text(json.dumps(protocol, indent=2) + "\n")
        print(f"VIVNet WLS: mode={'DDP' if distributed else 'single GPU'}, train={len(train_set)}, "
              f"val={len(val_set)}, GPUs={world_size}, "
              f"batch={args.batch_size_per_gpu} x {world_size}, crop={args.train_crop_size}", flush=True)

    iteration = start_iter
    epoch = start_epoch
    micro_size = args.micro_batch_size
    start_time = time.time()
    model.train()
    while iteration < args.total_iter:
        sampler.set_epoch(epoch)
        for batch_index, (inputs, targets) in enumerate(train_loader):
            if batch_index < start_batch:
                continue
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            slices = batch_micro_slices(inputs.shape[0], micro_size)
            batch_loss = 0.0
            for index, (chunk_start, chunk_end, chunk_weight) in enumerate(slices):
                micro_input = inputs[chunk_start:chunk_end]
                micro_target = targets[chunk_start:chunk_end]
                sync_context = (
                    model.no_sync()
                    if distributed and index + 1 < len(slices)
                    else nullcontext()
                )
                with sync_context:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                        prediction = model(micro_input)
                    prediction = prediction.float()
                    loss = F.l1_loss(prediction, micro_target.float()) + 0.1 * fft_loss(prediction, micro_target)
                    (loss * chunk_weight).backward()
                    batch_loss += loss.item() * chunk_weight
            optimizer.step()
            scheduler.step()
            iteration += 1
            next_epoch, next_batch = next_training_position(epoch, batch_index, len(train_loader))
            if rank == 0 and (iteration % 100 == 0 or iteration == start_iter + 1):
                print(f"iter={iteration}/{args.total_iter} epoch~{epoch} loss={batch_loss:.6f} "
                      f"lr={scheduler.get_last_lr()[0]:.4e} elapsed={time.time()-start_time:.1f}s", flush=True)

            if iteration % args.val_freq == 0 or iteration == args.total_iter:
                if rank == 0:
                    eval_model = model.module if distributed else model
                    metrics = evaluate(eval_model, val_loader, device)
                    if metrics["mse"] < best_val_mse:
                        best_val_mse = metrics["mse"]
                        save_checkpoint(best_path, model, optimizer, scheduler, iteration,
                                        next_epoch, next_batch, protocol, best_val_mse)
                    save_checkpoint(latest_path, model, optimizer, scheduler, iteration,
                                    next_epoch, next_batch, protocol, best_val_mse)
                    with (args.output_dir / "validation.csv").open("a", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=["iter", "epoch", *metrics.keys()])
                        if stream.tell() == 0:
                            writer.writeheader()
                        writer.writerow({"iter": iteration, "epoch": epoch, **metrics})
                    print(f"validation iter={iteration} metrics={metrics} best_val_mse={best_val_mse:.8f}", flush=True)
                if distributed:
                    dist.barrier()
            if iteration % args.save_freq == 0 or iteration == args.total_iter:
                if rank == 0:
                    save_checkpoint(latest_path, model, optimizer, scheduler, iteration,
                                    next_epoch, next_batch, protocol, best_val_mse)
                if distributed:
                    dist.barrier()
            if iteration >= args.total_iter:
                break
        epoch += 1
        start_batch = 0

    if rank == 0:
        print(f"Finished {iteration} updates in {time.time()-start_time:.1f}s", flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
