"""Windows single-process four-GPU DataParallel trainer for one-channel WLS."""

import argparse
from functools import partial
from itertools import islice
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader, Sampler, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.train_nafnet_wls import (
    autocast_context,
    build_run_protocol,
    evaluate,
    make_checkpoint,
    make_model_and_loss,
    save_checkpoint,
    seed_everything,
    split_microbatches,
    validate_resume_protocol,
)
from baseline.wls_data import WLSPairs, collate_pairs, paired_patch_collate, partition_ids


DEFAULT_DATA_ROOT = Path(r"D:\ResearchCode\S2R-HDR\data\20260818")
DEFAULT_OUTPUT_DIR = Path(
    r"D:\ResearchCode\halo-free\Decoupled_Learning\20260815_edgepreserving"
    r"\baseline\runs\nafnet_wls_patch256_4gpu"
)


class EpochRandomSampler(Sampler):
    """Replay the epoch order when resuming a checkpoint mid-epoch."""

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
    return gpu_count * per_gpu_batch, gpu_count * per_gpu_micro_batch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--total-iter", type=int, default=200000)
    parser.add_argument("--batch-size-per-gpu", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=8,
                        help="Micro-batch per GPU; 8 means one full forward per update")
    parser.add_argument("--train-crop-size", type=int, default=256)
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in yolo_clean")
    gpu_count = torch.cuda.device_count()
    if gpu_count != 4:
        raise RuntimeError(f"Expected four visible CUDA GPUs, found {gpu_count}")
    if args.train_crop_size < 0 or args.val_count < 1 or args.num_workers < 0:
        raise ValueError("Crop size and workers must be nonnegative; val-count must be positive")
    if min(args.total_iter, args.val_freq, args.save_freq, args.log_freq) < 1:
        raise ValueError("Iteration, validation, save, and log frequencies must be positive")
    global_batch, global_micro_batch = resolve_batch_sizes(
        gpu_count, args.batch_size_per_gpu, args.micro_batch_size
    )
    device = torch.device("cuda:0")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    all_train = WLSPairs(args.data_root, "train")
    train_ids, val_ids = partition_ids(all_train.ids, args.val_count, args.seed)
    id_to_index = {identifier: index for index, identifier in enumerate(all_train.ids)}
    train_set = Subset(all_train, [id_to_index[identifier] for identifier in train_ids])
    val_set = Subset(all_train, [id_to_index[identifier] for identifier in val_ids])
    protocol = build_run_protocol(
        args, train_ids, val_ids, world_size=gpu_count, parallelism="DataParallel"
    )

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

    base_model, criterion = make_model_and_loss(device)
    model = torch.nn.DataParallel(base_model, device_ids=list(range(gpu_count))).to(device)
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
    latest_path = args.output_dir / "nafnet_wls_latest.pth"
    best_path = args.output_dir / "nafnet_wls_best.pth"
    current_iter, epoch, next_batch, best_val = 0, 0, 0, float("inf")
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"No checkpoint to resume: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        validate_resume_protocol(checkpoint, protocol)
        model.module.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        current_iter = checkpoint["iter"]
        epoch = checkpoint["epoch"]
        next_batch = checkpoint["next_batch"]
        best_val = checkpoint["best_val_mse"]
    elif latest_path.exists() or best_path.exists():
        raise FileExistsError(f"Existing checkpoints in {args.output_dir}; use --resume or a new --output-dir")

    print(
        f"One-channel WLS NAFNet: DataParallel GPUs={gpu_count}, "
        f"batch/GPU={args.batch_size_per_gpu}, global batch={global_batch}, "
        f"micro-batch/GPU={args.micro_batch_size}, train={len(train_set)}, "
        f"validation={len(val_set)}, crop={args.train_crop_size or 'full-image'}, "
        f"updates={args.total_iter}, steps/epoch={len(train_loader)}",
        flush=True,
    )
    print(f"Protocol: {protocol}", flush=True)
    window_loss, window_steps = 0.0, 0
    started = time.time()
    while current_iter < args.total_iter:
        sampler.set_epoch(epoch)
        remaining = islice(train_loader, next_batch, None)
        progress = tqdm(
            remaining,
            total=len(train_loader),
            initial=next_batch,
            desc=f"Epoch {epoch + 1}",
            dynamic_ncols=True,
        )
        for batch_index, (images, targets) in enumerate(progress, start=next_batch):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            local_batch_size = images.shape[0]
            batch_loss = 0.0
            for micro_image, micro_target in split_microbatches(
                images, targets, global_micro_batch
            ):
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
            progress.set_postfix(
                iteration=f"{current_iter}/{args.total_iter}",
                loss=f"{batch_loss:.5f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            next_epoch, next_batch_index = epoch, batch_index + 1
            if next_batch_index >= len(train_loader):
                next_epoch, next_batch_index = epoch + 1, 0

            if current_iter % args.log_freq == 0:
                print(
                    f"iter={current_iter}/{args.total_iter} epoch~{epoch + 1} "
                    f"psnr_loss={window_loss / window_steps:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
                window_loss, window_steps = 0.0, 0

            if current_iter % args.save_freq == 0 or current_iter == args.total_iter:
                save_checkpoint(
                    latest_path,
                    make_checkpoint(
                        model, optimizer, scheduler, current_iter,
                        next_epoch, next_batch_index, best_val, protocol,
                    ),
                )

            if current_iter == 1000 or current_iter % args.val_freq == 0 or current_iter == args.total_iter:
                mse, mae = evaluate(model.module, val_loader, device, args.precision)
                print(f"iter={current_iter}: val MSE={mse:.8f}, val MAE={mae:.8f}", flush=True)
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

            epoch, next_batch = next_epoch, next_batch_index
            if current_iter >= args.total_iter:
                break


if __name__ == "__main__":
    main()
