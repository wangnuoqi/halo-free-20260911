"""Four-GPU DataParallel training for coarse-edge gated three-stage refinement."""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import CoarseEdgeGatedRefine3Stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--cache_manifest", default="manifest.jsonl")
    parser.add_argument("--output_dir", default="outputs/refine_3stage_exp1")
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--resize_scale", type=float, default=1.1)
    parser.add_argument("--disable_augment", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1, help="Global batch; raised to visible GPU count if smaller.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--resume", default=None, help="Checkpoint path, or 'latest' under --output_dir.")
    parser.add_argument("--l1_weight", type=float, default=0.5)
    parser.add_argument("--lpips_weight", type=float, default=0.25)
    parser.add_argument("--grad_weight", type=float, default=0.25)
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--residual_scale", type=float, default=0.20)
    parser.add_argument("--gate_beta", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260923)
    return parser.parse_args()


def visible_gpu_count() -> int:
    return max(1, len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]))


def _rng_state():
    import torch
    import numpy as np
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    import torch
    import numpy as np
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, scheduler, step, epoch, batch_in_epoch, args):
    import torch
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({
        "step": step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": vars(args),
        "rng_state": _rng_state(),
    }, path)


def main() -> None:
    import numpy as np
    import torch
    import torch.nn as nn
    import lpips
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from tqdm.auto import tqdm
    from refine_cache import CachedRefineDataset, load_cache_entries
    from train_refine_cosine import gradient_loss, l1_loss

    args = parse_args()
    print("[refine-3stage] initializing training", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Launch through wintrain/yolo_clean.")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    gpu_count = visible_gpu_count()
    batch_size = max(args.batch_size, gpu_count)
    dataset = CachedRefineDataset(load_cache_entries(args.cache_root, args.cache_manifest), args.resolution, args.resize_scale, args.disable_augment)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    if not len(loader):
        raise ValueError("No complete batch: lower --batch_size or provide more cached samples.")
    print(f"[refine-3stage] dataset ready: samples={len(dataset)} batches/epoch={len(loader)}", flush=True)
    model = CoarseEdgeGatedRefine3Stage(args.base_channels, args.num_blocks, args.residual_scale, args.gate_beta).cuda()
    if gpu_count > 1:
        model = nn.DataParallel(model)
    perceptual = lpips.LPIPS(net="alex").cuda().eval()
    for parameter in perceptual.parameters(): parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader), eta_min=args.min_learning_rate)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    writer, step = SummaryWriter(str(output_dir / "logs")), 0
    total_steps = args.epochs * len(loader)
    start_epoch, resume_batch = 0, 0
    resume_path = args.resume
    if resume_path == "latest":
        resume_path = str(output_dir / "checkpoint-latest.pth")
    if resume_path:
        checkpoint_path = Path(resume_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
        (model.module if hasattr(model, "module") else model).load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint.get("epoch", -1))
        resume_batch = int(checkpoint.get("batch_in_epoch", 0))
        if resume_batch >= len(loader):
            start_epoch += 1
            resume_batch = 0
        _restore_rng_state(checkpoint.get("rng_state"))
        print(f"Resumed from {checkpoint_path} | epoch={start_epoch + 1}/{args.epochs} | step={step}")
    print(f"Training examples: {len(dataset)} | GPUs: {gpu_count} | global batch: {batch_size}")
    print("[refine-3stage] entering training loop; first step appears after the first forward/backward pass", flush=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress = tqdm(enumerate(loader), total=len(loader), initial=resume_batch if epoch == start_epoch else 0,
                        desc=f"3stage exp1 {epoch + 1}/{args.epochs}", dynamic_ncols=True,
                        file=sys.stdout, mininterval=1.0, leave=True)
        last_loss = None
        for batch_idx, batch in progress:
            if epoch == start_epoch and batch_idx < resume_batch:
                continue
            if batch_idx == resume_batch:
                print(f"[refine-3stage] epoch {epoch + 1}: starting batch {batch_idx + 1}/{len(loader)} (global step {step + 1}/{total_steps})", flush=True)
            coarse, blended = batch["prelim"].cuda(non_blocking=True), batch["cond"].cuda(non_blocking=True)
            prior, target = batch["prior"].cuda(non_blocking=True), batch["gt"].cuda(non_blocking=True)
            output = model(coarse, blended, prior)["final"]
            l1 = l1_loss(output, target)
            lp = perceptual(output * 2 - 1, target * 2 - 1).mean()
            grad = gradient_loss(output, target)
            loss = args.l1_weight * l1 + args.lpips_weight * lp + args.grad_weight * grad
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); scheduler.step(); step += 1
            last_loss = loss.item()
            progress.set_postfix(step=f"{step}/{total_steps}", lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                                 loss=f"{last_loss:.4f}", l1=f"{l1.item():.4f}",
                                 lpips=f"{lp.item():.4f}", grad=f"{grad.item():.4f}")
            for name, value in (("loss", loss), ("l1", l1), ("lpips", lp), ("grad", grad)):
                writer.add_scalar(f"train/{name}", value.item(), step)
            if args.checkpointing_steps and step % args.checkpointing_steps == 0:
                save_checkpoint(output_dir / f"checkpoint-{step}.pth", model, optimizer, scheduler, step, epoch, batch_idx + 1, args)
                save_checkpoint(output_dir / "checkpoint-latest.pth", model, optimizer, scheduler, step, epoch, batch_idx + 1, args)
                numbered = [p for p in output_dir.glob("checkpoint-*.pth") if p.stem.split("-")[-1].isdigit()]
                old = sorted(numbered, key=lambda p: int(p.stem.split("-")[-1]))[:-args.checkpoints_total_limit]
                for path in old: path.unlink()
        resume_batch = 0
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={(last_loss if last_loss is not None else float('nan')):.5f}")
        save_checkpoint(output_dir / "checkpoint-latest.pth", model, optimizer, scheduler, step, epoch, len(loader), args)
    save_checkpoint(output_dir / "refine_3stage_exp1_final.pth", model, optimizer, scheduler, step, args.epochs - 1, len(loader), args)
    writer.close()


if __name__ == "__main__":
    main()
