"""Windows single-process multi-GPU training for cached NAF refinement."""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path


def visible_gpu_count(cuda_visible_devices: str | None) -> int:
    devices = [
        item.strip()
        for item in (cuda_visible_devices or "").split(",")
        if item.strip()
    ]
    return max(1, len(devices))


def effective_batch_size(requested: int, cuda_visible_devices: str | None) -> int:
    if requested <= 0:
        raise ValueError("batch_size must be positive")
    return max(requested, visible_gpu_count(cuda_visible_devices))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train cached NAF refinement with Windows DataParallel."
    )
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--cache_manifest", default="manifest.jsonl")
    parser.add_argument("--output_dir", default="outputs/naf_dataparallel")
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--resize_scale", type=float, default=1.1)
    parser.add_argument("--disable_augment", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--l1_weight", type=float, default=0.5)
    parser.add_argument("--lpips_weight", type=float, default=0.25)
    parser.add_argument("--grad_weight", type=float, default=0.25)
    parser.add_argument("--nafnet_width", type=int, default=64)
    parser.add_argument("--nafnet_middle_blk_num", type=int, default=1)
    parser.add_argument("--nafnet_enc_blk_nums", type=int, nargs="+", default=[1, 1, 1, 28])
    parser.add_argument("--nafnet_dec_blk_nums", type=int, nargs="+", default=[1, 1, 1, 1])
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _trim_old_checkpoints(output_dir: Path, limit: int) -> None:
    if limit is None or limit <= 0:
        return
    checkpoints = sorted(
        output_dir.glob("checkpoint-*.pth"),
        key=lambda path: int(path.stem.rsplit("-", 1)[1]),
    )
    for path in checkpoints[:-limit]:
        path.unlink(missing_ok=True)


def main() -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from tqdm.auto import tqdm
    import lpips

    from refine_cache import CachedRefineDataset, load_cache_entries
    from train_refine_cosine import (
        compute_hf_image,
        gradient_loss,
        l1_loss,
    )
    from naf_loader import load_nafnet_class

    args = parse_args()
    if len(args.nafnet_enc_blk_nums) != len(args.nafnet_dec_blk_nums):
        raise ValueError("Encoder and decoder block lists must have the same length.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Windows DataParallel trainer.")

    _set_seed(args.seed)
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_count = visible_gpu_count(cuda_devices)
    batch_size = effective_batch_size(args.batch_size, cuda_devices)
    device = torch.device("cuda:0")

    entries = load_cache_entries(args.cache_root, args.cache_manifest)
    dataset = CachedRefineDataset(
        entries,
        args.resolution,
        args.resize_scale,
        args.disable_augment,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    NAFNet = load_nafnet_class()

    class RefineModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.naf = NAFNet(
                img_channel=10,
                width=args.nafnet_width,
                middle_blk_num=args.nafnet_middle_blk_num,
                enc_blk_nums=args.nafnet_enc_blk_nums,
                dec_blk_nums=args.nafnet_dec_blk_nums,
            )
            self.head = nn.Conv2d(10, 3, kernel_size=1, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.naf(x)).clamp(0.0, 1.0)

    model = RefineModel().to(device)
    if gpu_count > 1:
        model = nn.DataParallel(model)

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps),
        eta_min=args.min_learning_rate,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(output_dir / "logs"))
    global_step = 0

    print(f"Training examples: {len(dataset)}")
    print(f"Visible GPUs: {gpu_count}")
    print(f"DataParallel batch size: {batch_size}")
    print(f"Batches per epoch: {len(loader)}")

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(
            loader,
            total=len(loader),
            desc=f"NAF train {epoch + 1}/{args.epochs}",
            dynamic_ncols=True,
        )
        for batch in progress:
            cond = batch["cond"].to(device, non_blocking=True)
            coarse = batch["prelim"].to(device, non_blocking=True)
            prior = batch["prior"].to(device, non_blocking=True)
            target = batch["gt"].to(device, non_blocking=True)

            with torch.no_grad():
                high_frequency = compute_hf_image(cond)
            model_input = torch.cat([coarse, high_frequency, prior, cond], dim=1)

            prediction = model(model_input)
            l1_value = l1_loss(prediction, target)
            lpips_value = lpips_model(
                prediction * 2.0 - 1.0,
                target * 2.0 - 1.0,
            ).mean()
            grad_value = gradient_loss(prediction, target)
            loss = (
                args.l1_weight * l1_value
                + args.lpips_weight * lpips_value
                + args.grad_weight * grad_value
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/l1", l1_value.item(), global_step)
            writer.add_scalar("train/lpips", lpips_value.item(), global_step)
            writer.add_scalar("train/grad", grad_value.item(), global_step)
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "model_state_dict": state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    output_dir / f"checkpoint-{global_step}.pth",
                )
                _trim_old_checkpoints(output_dir, args.checkpoints_total_limit)

        print(f"Epoch {epoch + 1}/{args.epochs} complete")

    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state, output_dir / "nafnet_refine_final.pth")
    writer.close()
    print(f"Saved final model: {output_dir / 'nafnet_refine_final.pth'}")


if __name__ == "__main__":
    main()
