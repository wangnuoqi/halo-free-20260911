"""Windows single-process four-GPU DataParallel trainer for WLS VIVNet."""

import argparse
import csv
from functools import partial
from itertools import islice
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baseline.wls_data import WLSPairs, collate_pairs, paired_patch_collate, partition_ids
from baseline.vivnet_wls.model import build_model, run_padded
from baseline.vivnet_wls.train import (
    batch_micro_slices,
    fft_loss,
    next_training_position,
    protocol_for,
    validate_resume_protocol,
)


DEFAULT_DATA_ROOT = Path(r"D:\ResearchCode\S2R-HDR\data\20260818")
DEFAULT_OUTPUT_DIR = Path(
    r"D:\ResearchCode\halo-free\Decoupled_Learning\20260815_edgepreserving"
    r"\baseline\runs\vivnet_wls_patch256_wintrain_4gpu_bs64"
)


class EpochRandomSampler(Sampler):
    """Replay a deterministic shuffle order when resuming within an epoch."""

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self):
        return len(self.dataset)


def resolve_batch_sizes(gpu_count, per_gpu_batch, per_gpu_micro_batch):
    if gpu_count < 1 or per_gpu_batch < 1 or not 1 <= per_gpu_micro_batch <= per_gpu_batch:
        raise ValueError("Invalid GPU, batch, or micro-batch count")
    if per_gpu_batch % per_gpu_micro_batch:
        raise ValueError("Per-GPU batch must be divisible by its micro-batch")
    return gpu_count * per_gpu_batch, gpu_count * per_gpu_micro_batch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--total-iter", type=int, default=75000)
    parser.add_argument("--batch-size-per-gpu", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--train-crop-size", type=int, default=256)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--val-freq", type=int, default=15000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def autocast_context(device, precision):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=precision == "bf16",
    )


def save_checkpoint(path, model, optimizer, scheduler, iteration, epoch, next_batch,
                    protocol, best_val_mse):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "iter": iteration,
        "epoch": epoch,
        "next_batch": next_batch,
        "best_val_mse": best_val_mse,
        "run_protocol": protocol,
    }, path)


def evaluate(model, loader, device, precision):
    rows = []
    model.eval()
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast_context(device, precision):
                prediction = run_padded(model, inputs)
            error = prediction.float().clamp(0.0, 1.0) - targets.float()
            mse = error.square().flatten(1).mean(1)
            mae = error.abs().flatten(1).mean(1)
            rows.extend((float(m), float(a), float(-10.0 * torch.log10(m.clamp_min(1e-12))))
                        for m, a in zip(mse.cpu(), mae.cpu()))
    model.train()
    return {
        "images": len(rows),
        "mse": float(np.mean([row[0] for row in rows])),
        "mae": float(np.mean([row[1] for row in rows])),
        "mean_psnr_db": float(np.mean([row[2] for row in rows])),
    }


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in yolo_clean")
    gpu_count = torch.cuda.device_count()
    if gpu_count != 4:
        raise RuntimeError(f"Expected four visible CUDA GPUs, found {gpu_count}")
    if args.val_count < 1 or args.num_workers < 0 or args.train_crop_size < 0:
        raise ValueError("val-count must be positive; workers and crop size must be nonnegative")
    if min(args.total_iter, args.val_freq, args.save_freq, args.log_freq) < 1:
        raise ValueError("Iteration and logging frequencies must be positive")
    global_batch, global_micro_batch = resolve_batch_sizes(
        gpu_count, args.batch_size_per_gpu, args.micro_batch_size
    )

    device = torch.device("cuda:0")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    all_train = WLSPairs(args.data_root, "train")
    train_ids, val_ids = partition_ids(all_train.ids, args.val_count, args.seed)
    id_to_index = {identifier: index for index, identifier in enumerate(all_train.ids)}
    train_set = Subset(all_train, [id_to_index[key] for key in train_ids])
    val_set = Subset(all_train, [id_to_index[key] for key in val_ids])
    protocol = protocol_for(args, gpu_count, train_ids, val_ids)
    protocol["parallelism"] = "DataParallel"
    protocol["global_micro_batch_size"] = global_micro_batch

    sampler = EpochRandomSampler(train_set, args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=global_batch,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        collate_fn=partial(paired_patch_collate, crop_size=args.train_crop_size),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_pairs,
    )

    model = torch.nn.DataParallel(build_model(device), device_ids=list(range(gpu_count)))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_iter, eta_min=args.min_lr
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.output_dir / "vivnet_wls_latest.pth"
    best_path = args.output_dir / "vivnet_wls_best.pth"
    iteration, epoch, next_batch = 0, 0, 0
    best_val_mse = float("inf")
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"No checkpoint to resume: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        validate_resume_protocol(checkpoint, protocol)
        model.module.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        iteration = int(checkpoint["iter"])
        epoch = int(checkpoint["epoch"])
        next_batch = int(checkpoint["next_batch"])
        best_val_mse = float(checkpoint["best_val_mse"])
    elif latest_path.exists() or best_path.exists():
        raise FileExistsError(f"Existing checkpoints in {args.output_dir}; use --resume or a new output directory")

    protocol_path = args.output_dir / "config_used.json"
    protocol_path.write_text(__import__("json").dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(
        f"VIVNet WLS DataParallel: GPUs={gpu_count}, batch/GPU={args.batch_size_per_gpu}, "
        f"global batch={global_batch}, micro/GPU={args.micro_batch_size}, "
        f"train={len(train_set)}, val={len(val_set)}, crop={args.train_crop_size}, "
        f"updates={args.total_iter}, steps/epoch={len(train_loader)}",
        flush=True,
    )
    print(f"Protocol: {protocol}", flush=True)

    window_loss, window_steps = 0.0, 0
    started = time.time()
    while iteration < args.total_iter:
        sampler.set_epoch(epoch)
        progress = tqdm(
            islice(train_loader, next_batch, None),
            total=len(train_loader), initial=next_batch,
            desc=f"Epoch {epoch + 1}", dynamic_ncols=True,
        )
        for batch_index, (images, targets) in enumerate(progress, start=next_batch):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            local_batch_size = images.shape[0]
            batch_loss = 0.0
            slices = batch_micro_slices(local_batch_size, global_micro_batch)
            for chunk_start, chunk_end, chunk_weight in slices:
                micro_images = images[chunk_start:chunk_end].to(device, non_blocking=True)
                micro_targets = targets[chunk_start:chunk_end].to(device, non_blocking=True)
                with autocast_context(device, args.precision):
                    prediction = model(micro_images)
                loss = F.l1_loss(prediction.float(), micro_targets.float()) + 0.1 * fft_loss(
                    prediction.float(), micro_targets.float()
                )
                (loss * chunk_weight).backward()
                batch_loss += loss.detach().item() * chunk_weight
            optimizer.step()
            scheduler.step()
            iteration += 1
            window_loss += batch_loss
            window_steps += 1
            progress.set_postfix(
                iteration=f"{iteration}/{args.total_iter}",
                loss=f"{batch_loss:.5f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            next_epoch, next_batch_index = next_training_position(epoch, batch_index, len(train_loader))

            if iteration % args.log_freq == 0:
                print(
                    f"iter={iteration}/{args.total_iter} epoch~{epoch + 1} "
                    f"loss={window_loss / window_steps:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.4e} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
                window_loss, window_steps = 0.0, 0

            if iteration % args.val_freq == 0 or iteration == args.total_iter:
                metrics = evaluate(model, val_loader, device, args.precision)
                if metrics["mse"] < best_val_mse:
                    best_val_mse = metrics["mse"]
                    save_checkpoint(best_path, model, optimizer, scheduler, iteration,
                                    next_epoch, next_batch_index, protocol, best_val_mse)
                save_checkpoint(latest_path, model, optimizer, scheduler, iteration,
                                next_epoch, next_batch_index, protocol, best_val_mse)
                with (args.output_dir / "validation.csv").open("a", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["iter", "epoch", *metrics.keys()])
                    if stream.tell() == 0:
                        writer.writeheader()
                    writer.writerow({"iter": iteration, "epoch": next_epoch, **metrics})
                print(f"validation iter={iteration} metrics={metrics} best_val_mse={best_val_mse:.8f}", flush=True)
            if iteration % args.save_freq == 0 or iteration == args.total_iter:
                save_checkpoint(latest_path, model, optimizer, scheduler, iteration,
                                next_epoch, next_batch_index, protocol, best_val_mse)
            if iteration >= args.total_iter:
                break
        epoch += 1
        next_batch = 0

    print(f"Finished {iteration} updates in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
