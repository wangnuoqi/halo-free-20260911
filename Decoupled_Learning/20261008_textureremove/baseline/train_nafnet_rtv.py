"""Train NAFNet on paired RTV images with NCCL DDP and full-image validation."""

import argparse
from contextlib import nullcontext
from functools import partial
from itertools import islice
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAFNET_ROOT = Path(__file__).resolve().parent / "NAFNet"
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent.parent / "datasets" / "VOC2012" / "RTV_RGB_fixed_lambda_0p01"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runs" / "nafnet_rtv_patch256_3gpu"

sys.path.insert(0, str(PROJECT_ROOT))

from rtv_training_common import RTVDataset, collate_pairs, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train NAFNet on paired RTV TIFFs using NCCL DDP; validate on full images."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--total-iter", type=int, default=200000)
    parser.add_argument("--batch-size-per-gpu", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--train-crop-size", type=int, default=256,
                        help="Random paired training crop size; set 0 for batch-aligned full images")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--val-freq", type=int, default=20000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_run_protocol(args):
    world_size = int(os.environ.get("WORLD_SIZE", "3"))
    crop_size = args.train_crop_size
    return {
        "architecture": "NAFNetLocal-width32-gopro",
        "train_geometry": "random_paired_patch" if crop_size else "batch_min_top_left_even_full_image",
        "train_crop_size": crop_size,
        "validation_geometry": "full_image_even",
        "channel_order": "BGR",
        "value_range": "0_1",
        "world_size": world_size,
        "batch_size_per_gpu": args.batch_size_per_gpu,
        "effective_batch_size": world_size * args.batch_size_per_gpu,
        "micro_batch_size": args.micro_batch_size,
        "pin_memory": False,
        "precision": args.precision,
        "total_iter": args.total_iter,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "betas": [args.beta1, args.beta2],
        "scheduler": "CosineAnnealingLR",
        "min_lr": args.min_lr,
        "loss": "PSNRLoss",
        "val_freq": args.val_freq,
        "save_freq": args.save_freq,
        "seed": args.seed,
    }


def validate_resume_protocol(checkpoint, current_protocol):
    saved_protocol = checkpoint.get("run_protocol")
    if saved_protocol is None:
        raise ValueError("Checkpoint does not contain a run protocol")
    saved_protocol = dict(saved_protocol)
    requested_protocol = dict(current_protocol)
    # Microbatch size changes only how the same per-GPU batch is partitioned
    # for memory use; the weighted accumulated gradient and optimizer updates
    # remain the same.
    saved_protocol.pop("micro_batch_size", None)
    requested_protocol.pop("micro_batch_size", None)
    if saved_protocol != requested_protocol:
        raise ValueError(
            "Checkpoint training protocol does not match current arguments; "
            "resume with the original data geometry, GPU count, and optimizer settings"
        )


def split_microbatches(inputs, targets, micro_batch_size):
    if micro_batch_size < 1:
        raise ValueError("micro_batch_size must be positive")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("Input and target batch sizes must match")
    return [
        (inputs[start : start + micro_batch_size], targets[start : start + micro_batch_size])
        for start in range(0, inputs.shape[0], micro_batch_size)
    ]


def paired_patch_collate(batch, crop_size):
    if crop_size == 0:
        return collate_pairs(batch)
    if crop_size < 0:
        raise ValueError("train_crop_size must be zero or positive")

    images = []
    targets = []
    for image, target in batch:
        if image.shape != target.shape:
            raise ValueError(f"Paired input/target shape mismatch: {image.shape} vs {target.shape}")
        height, width = image.shape[-2:]
        if height < crop_size:
            rows = torch.arange(crop_size, device=image.device).remainder(2 * height)
            rows = torch.where(rows < height, rows, 2 * height - 1 - rows)
            image = image.index_select(-2, rows)
            target = target.index_select(-2, rows)
            height = crop_size
        if width < crop_size:
            cols = torch.arange(crop_size, device=image.device).remainder(2 * width)
            cols = torch.where(cols < width, cols, 2 * width - 1 - cols)
            image = image.index_select(-1, cols)
            target = target.index_select(-1, cols)
            width = crop_size
        top = int(torch.randint(height - crop_size + 1, ()).item())
        left = int(torch.randint(width - crop_size + 1, ()).item())
        images.append(image[:, top : top + crop_size, left : left + crop_size])
        targets.append(target[:, top : top + crop_size, left : left + crop_size])
    return torch.stack(images), torch.stack(targets)


def make_model_and_loss(device):
    sys.path.insert(0, str(NAFNET_ROOT))
    from basicsr.models.archs.NAFNet_arch import NAFNetLocal
    from basicsr.models.losses.losses import PSNRLoss

    model = NAFNetLocal(
        img_channel=3,
        width=32,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
        train_size=(1, 3, 256, 256),
    ).to(device)
    return model, PSNRLoss(loss_weight=1.0, reduction="mean").to(device)


def autocast_context(device, precision):
    enabled = precision == "bf16"
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=enabled,
    )


def evaluate(model, loader, device, precision):
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    with torch.no_grad():
        for image, target in loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with autocast_context(device, precision):
                prediction = model(image)
            prediction = prediction.float().clamp(0.0, 1.0)
            target = target.float()
            mse_sum += torch.mean((prediction - target) ** 2).item()
            mae_sum += torch.mean((prediction - target).abs()).item()
    model.train()
    return mse_sum / max(1, len(loader)), mae_sum / max(1, len(loader))


def save_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp_path)
    os.replace(temp_path, path)


def make_checkpoint(model, optimizer, scheduler, current_iter, epoch, next_batch,
                    best_val, protocol):
    return {
        "iter": current_iter,
        "epoch": epoch,
        "next_batch": next_batch,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_mse": best_val,
        "run_protocol": protocol,
    }


def main():
    args = parse_args()
    if "LOCAL_RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Launch this script with torchrun --nproc_per_node=3")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 3:
        raise RuntimeError(f"Expected exactly 3 DDP processes, got {world_size}")
    if args.batch_size_per_gpu < 1 or args.micro_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.train_crop_size < 0:
        raise ValueError("train-crop-size must be zero or positive")
    if args.micro_batch_size > args.batch_size_per_gpu:
        raise ValueError("micro-batch-size cannot exceed batch-size-per-gpu")
    if min(args.total_iter, args.val_freq, args.save_freq, args.log_freq) < 1:
        raise ValueError("Iteration and logging frequencies must be positive")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", local_rank)
    try:
        seed_everything(args.seed + rank)
        torch.backends.cudnn.benchmark = True
        protocol = build_run_protocol(args)

        train_set = RTVDataset(str(args.data_root / "train"))
        val_set = RTVDataset(str(args.data_root / "val")) if rank == 0 else None
        sampler = DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=True,
            drop_last=False, seed=args.seed,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size_per_gpu,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=False,
            persistent_workers=args.num_workers > 0,
            drop_last=False,
            collate_fn=partial(paired_patch_collate, crop_size=args.train_crop_size),
        )
        val_loader = None
        if rank == 0:
            val_loader = DataLoader(
                val_set,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=False,
                persistent_workers=args.num_workers > 0,
                collate_fn=collate_pairs,
            )

        model, criterion = make_model_and_loss(device)
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.total_iter, eta_min=args.min_lr
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        latest_path = args.output_dir / "nafnet_rtv_latest.pth"
        best_path = args.output_dir / "nafnet_rtv_best.pth"
        current_iter = 0
        epoch = 0
        next_batch = 0
        best_val = float("inf")
        if args.resume:
            if not latest_path.is_file():
                raise FileNotFoundError(f"No checkpoint to resume: {latest_path}")
            checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
            validate_resume_protocol(checkpoint, protocol)
            model.module.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            current_iter = checkpoint["iter"]
            epoch = checkpoint["epoch"]
            next_batch = checkpoint["next_batch"]
            best_val = checkpoint["best_val_mse"]

        if rank == 0:
            print(
                f"NAFNet RTV full-image DDP: GPUs={world_size}, "
                f"per-GPU batch={args.batch_size_per_gpu}, "
                f"effective batch={protocol['effective_batch_size']}, "
                f"micro-batch={args.micro_batch_size}, "
                f"train-crop={args.train_crop_size or 'full-image'}, "
                f"iterations={args.total_iter}, loader steps/epoch={len(train_loader)}"
            )
            print(f"Protocol: {protocol}")

        window_loss = 0.0
        window_steps = 0
        start_time = time.time()
        while current_iter < args.total_iter:
            sampler.set_epoch(epoch)
            remaining_batches = islice(train_loader, next_batch, None)
            epoch_loader = remaining_batches
            if rank == 0:
                epoch_loader = tqdm(
                    remaining_batches,
                    total=len(train_loader),
                    initial=next_batch,
                    desc=f"Epoch {epoch + 1}",
                    dynamic_ncols=True,
                )
            for batch_index, (images, targets) in enumerate(epoch_loader, start=next_batch):
                if current_iter >= args.total_iter:
                    break

                model.train()
                optimizer.zero_grad(set_to_none=True)
                microbatches = split_microbatches(images, targets, args.micro_batch_size)
                local_batch_size = images.shape[0]
                batch_loss = 0.0
                for micro_index, (micro_image, micro_target) in enumerate(microbatches):
                    sync_context = (
                        model.no_sync()
                        if micro_index < len(microbatches) - 1
                        else nullcontext()
                    )
                    with sync_context:
                        micro_image = micro_image.to(device, non_blocking=True)
                        micro_target = micro_target.to(device, non_blocking=True)
                        with autocast_context(device, args.precision):
                            prediction = model(micro_image)
                        loss = criterion(prediction.float(), micro_target.float())
                        scaled_loss = loss * (micro_image.shape[0] / local_batch_size)
                        scaled_loss.backward()
                    batch_loss += scaled_loss.detach().item()

                optimizer.step()
                scheduler.step()
                current_iter += 1
                window_loss += batch_loss
                window_steps += 1
                if rank == 0:
                    epoch_loader.set_postfix(
                        iteration=f"{current_iter}/{args.total_iter}",
                        loss=f"{batch_loss:.5f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )
                next_epoch, next_batch_index = epoch, batch_index + 1
                if next_batch_index >= len(train_loader):
                    next_epoch, next_batch_index = epoch + 1, 0

                if current_iter % args.log_freq == 0:
                    loss_tensor = torch.tensor(
                        [window_loss, window_steps], device=device, dtype=torch.float64
                    )
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    if rank == 0:
                        mean_loss = (loss_tensor[0] / loss_tensor[1]).item()
                        lr_now = optimizer.param_groups[0]["lr"]
                        print(
                            f"iter={current_iter}/{args.total_iter} "
                            f"epoch~{epoch + 1} psnr_loss={mean_loss:.6f} "
                            f"lr={lr_now:.3e} elapsed={time.time() - start_time:.1f}s"
                        )
                    window_loss = 0.0
                    window_steps = 0

                if current_iter % args.save_freq == 0 or current_iter == args.total_iter:
                    if rank == 0:
                        save_checkpoint(
                            latest_path,
                            make_checkpoint(
                                model, optimizer, scheduler, current_iter,
                                next_epoch, next_batch_index, best_val, protocol,
                            ),
                        )

                should_validate = (
                    current_iter == 1000
                    or current_iter % args.val_freq == 0
                    or current_iter == args.total_iter
                )
                if should_validate:
                    dist.barrier()
                    if rank == 0:
                        mse, mae = evaluate(model.module, val_loader, device, args.precision)
                        print(
                            f"iter={current_iter}: val MSE={mse:.8f}, "
                            f"val MAE={mae:.8f}"
                        )
                        if mse < best_val:
                            best_val = mse
                            save_checkpoint(
                                best_path,
                                make_checkpoint(
                                    model, optimizer, scheduler, current_iter,
                                    next_epoch, next_batch_index, best_val, protocol,
                                ),
                            )
                        save_checkpoint(
                            latest_path,
                            make_checkpoint(
                                model, optimizer, scheduler, current_iter,
                                next_epoch, next_batch_index, best_val, protocol,
                            ),
                        )
                    dist.barrier()

                epoch, next_batch = next_epoch, next_batch_index
            else:
                continue
            break
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
