"""Matched patch training for an E0/E1 architecture comparison.

The only intended difference between ``legacy_matched_e0`` and
``legacy_matched_e1`` is the coarse LR predictor. Both runs use the historical
Exp7C optimization recipe while selecting checkpoints on an internal
validation split rather than on test images.
    """
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

# Allow both ``python -m expe.train_legacy_matched`` and direct execution from
# the experiment directory, where Python otherwise only adds ``expe/`` to the
# import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp_b.src.data_exp_b import (
    PairedPatchCollator,
    PairedTripletDataset,
    deterministic_train_val_indices,
    seed_worker,
)
from exp_b.src.train_exp_b import delta_cascade_loss, edge_guided_loss
from exp_b.src.model_exp_b import ThreeStageRefine
from expe.model import build_expe_model


def init_dist(parallelism: str, device_arg: str):
    if parallelism == "dp":
        if not torch.cuda.is_available():
            raise RuntimeError("DataParallel matched training requires CUDA")
        return False, 0, torch.cuda.device_count(), torch.device("cuda", 0)
    if parallelism == "single":
        if device_arg == "cuda" or (device_arg == "auto" and torch.cuda.is_available()):
            return False, 0, 1, torch.device("cuda", 0)
        return False, 0, 1, torch.device("cpu")
    if "LOCAL_RANK" not in os.environ:
        if device_arg == "cuda" and torch.cuda.is_available():
            return False, 0, 1, torch.device("cuda", 0)
        return False, 0, 1, torch.device("cpu")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    visible_devices = torch.cuda.device_count()
    if visible_devices < world:
        raise RuntimeError(
            f"requested {world} ranks, but only {visible_devices} CUDA devices are visible"
        )
    torch.cuda.set_device(local_rank)
    backend = os.environ.get("TORCH_DDP_BACKEND", "nccl").lower()
    if backend not in {"nccl", "gloo"}:
        raise ValueError(f"unsupported DDP backend: {backend}")
    dist.init_process_group(backend, init_method="env://")
    return True, rank, world, torch.device("cuda", local_rank)


def seed_all(seed: int, rank: int):
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(value)


class MatchedModel(torch.nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        if variant not in {"legacy_matched_e0", "legacy_matched_e1"}:
            raise ValueError(f"unknown matched variant: {variant}")
        coarse_variant = "exp7c_e0" if variant.endswith("e0") else "viv_gf_e1"
        self.variant = variant
        self.coarse = build_expe_model(coarse_variant)
        self.refine = ThreeStageRefine()
        self.phase = "joint"

    def forward(self, lr: torch.Tensor, hr: torch.Tensor):
        # Both public coarse models return raw [0,1] images.
        coarse0 = self.coarse(hr, lr) - 0.5
        if self.phase == "base":
            return {"coarse0": coarse0}
        refined = self.refine(hr - 0.5, coarse0)
        return {"coarse0": coarse0, **refined}


def matched_loss(outputs, target_raw: torch.Tensor):
    target = target_raw - 0.5
    delta_gt = target - outputs["coarse0"].float()
    l1 = edge_guided_loss(outputs["delta1"], delta_gt)
    l2 = edge_guided_loss(outputs["delta2"], delta_gt)
    l3 = edge_guided_loss(outputs["delta3"], delta_gt)
    cascade = 0.1 * l1 + 0.3 * l2 + 0.6 * l3
    coarse = F.smooth_l1_loss(outputs["coarse0"].float(), target.float())
    return cascade + 0.5 * coarse


def phase_loss(outputs, target_raw: torch.Tensor, phase: str):
    if phase == "base":
        prediction = outputs["coarse0"].float()
        target = (target_raw - 0.5).float()
        pixel = F.mse_loss(prediction, target)
        kx = prediction.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))).view(1, 1, 3, 3)
        ky = prediction.new_tensor(((-1, -2, -1), (0, 0, 0), (1, 2, 1))).view(1, 1, 3, 3)
        pred_pad = F.pad(prediction, (1, 1, 1, 1), mode="reflect")
        target_pad = F.pad(target, (1, 1, 1, 1), mode="reflect")
        grad = ((F.conv2d(pred_pad, kx) - F.conv2d(target_pad, kx)).abs() +
                (F.conv2d(pred_pad, ky) - F.conv2d(target_pad, ky)).abs()).mean()
        grad_scale = (grad / (pixel + 1e-8)).detach().clamp_min(1e-8)
        return pixel + grad / grad_scale
    cascade, _ = delta_cascade_loss(outputs, target_raw.float() - 0.5)
    if phase == "refine":
        return cascade
    if phase == "joint":
        return cascade + 0.5 * F.smooth_l1_loss(
            outputs["coarse0"].float(), (target_raw - 0.5).float()
        )
    raise ValueError(f"unknown phase: {phase}")


def phase_learning_rates(phase: str, args):
    if phase == "base":
        return [("coarse", args.lr_base)]
    if phase == "refine":
        return [("refine", args.lr_refine)]
    if phase == "joint":
        return [("coarse", args.lr_joint_coarse), ("refine", args.lr_joint_refine)]
    raise ValueError(f"unknown phase: {phase}")


def metric(outputs, gt_raw):
    pred_key = "coarse0" if outputs.get("_phase") == "base" else "final"
    pred = (outputs[pred_key] + 0.5).float().clamp(0.0, 1.0)
    target = gt_raw.float()
    mse = F.mse_loss(pred, target).item()
    return mse, 10.0 * math.log10(1.0 / max(mse, 1e-12))


def reduce_pair(device, total, count):
    values = torch.tensor([total, count], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(values[0].item()), float(values[1].item())


def save_checkpoint(path, model, optimizer, scheduler, iteration, best_mse, protocol):
    unwrapped = model.module if isinstance(model, (DDP, DataParallel)) else model
    torch.save(
        {
            "variant": protocol["variant"],
            "phase": protocol["phase"],
            "implementation_revision": 5,
            "iteration": iteration,
            "model_state": unwrapped.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_mse": best_mse,
            "protocol": protocol,
        },
        path,
    )


def check_parent_protocol(state, args, world, train_ids, val_ids):
    parent = state.get("protocol", {})
    expected = {
        "data_root": str(Path(args.data_root).resolve()),
        "train_geometry": f"random_aligned_{args.crop_size}_hr_{args.crop_size // 2}_lr_patch",
        "global_batch_size": args.batch_size_per_rank * world * args.grad_accumulation,
        "total_iter": args.total_iter,
        "train_ids": train_ids,
        "validation_ids": val_ids,
    }
    if any(parent.get(key) != value for key, value in expected.items()):
        raise ValueError("Parent checkpoint does not match the exposure/data protocol")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("legacy_matched_e0", "legacy_matched_e1"))
    parser.add_argument("--phase", required=True, choices=("base", "refine", "joint"))
    parser.add_argument("--parallelism", choices=("dp", "ddp", "single"), default="dp")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-iter", type=int, default=60000)
    parser.add_argument("--val-every", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--batch-size-per-rank", type=int, default=16)
    parser.add_argument("--grad-accumulation", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-base", type=float, default=1e-4)
    parser.add_argument("--lr-refine", type=float, default=1e-2)
    parser.add_argument("--lr-joint-coarse", type=float, default=3e-4)
    parser.add_argument("--lr-joint-refine", type=float, default=3e-3)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--refine-checkpoint", default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if min(args.total_iter, args.val_every, args.save_every, args.grad_accumulation) < 1:
        parser.error("total-iter, val-every, save-every and grad-accumulation must be positive")

    distributed, rank, world, device = init_dist(args.parallelism, args.device)
    is_main = rank == 0
    seed_all(args.seed, rank)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.pth"
    best = output / "best.pth"
    final = output / "final.pth"
    log_path = output / "history.jsonl"
    if not args.resume and any(p.exists() for p in (latest, best, final, log_path)):
        raise FileExistsError(f"existing matched run in {output}; use --resume or a new output")

    raw = PairedTripletDataset(args.data_root, split="train")
    train_ids, val_ids = deterministic_train_val_indices(raw.ids, args.val_count, args.seed)
    train_set, val_set = Subset(raw, train_ids), Subset(raw, val_ids)
    sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if distributed else None
    loader_batch_size = args.batch_size_per_rank * (torch.cuda.device_count() if args.parallelism == "dp" else 1)
    train_loader = DataLoader(
        train_set,
        batch_size=loader_batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        collate_fn=PairedPatchCollator(crop_size=args.crop_size, random_crop=True),
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )
    val_sampler = DistributedSampler(val_set, num_replicas=world, rank=rank, shuffle=False) if distributed else None
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=PairedPatchCollator(crop_size=0, random_crop=False),
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )

    model = MatchedModel(args.variant).to(device)
    model.phase = args.phase
    if args.phase in {"refine", "joint"}:
        if not args.base_checkpoint:
            raise ValueError(f"--base-checkpoint is required for phase={args.phase}")
        base_state = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
        if base_state.get("variant") != args.variant or base_state.get("phase") != "base":
            raise ValueError("base checkpoint does not match variant/phase")
        if base_state.get("implementation_revision") != 5:
            raise ValueError("Base checkpoint uses an obsolete implementation/protocol; retrain in a new run directory")
        check_parent_protocol(base_state, args, world, [raw.ids[i] for i in train_ids],
                              [raw.ids[i] for i in val_ids])
        model.load_state_dict(base_state["model_state"], strict=True)
    if args.phase == "joint":
        if not args.refine_checkpoint:
            raise ValueError("--refine-checkpoint is required for phase=joint")
        refine_state = torch.load(args.refine_checkpoint, map_location="cpu", weights_only=False)
        if refine_state.get("variant") != args.variant or refine_state.get("phase") != "refine":
            raise ValueError("refine checkpoint does not match variant/phase")
        if refine_state.get("implementation_revision") != 5:
            raise ValueError("Refine checkpoint uses an obsolete implementation/protocol")
        check_parent_protocol(refine_state, args, world, [raw.ids[i] for i in train_ids],
                              [raw.ids[i] for i in val_ids])
        model.load_state_dict(refine_state["model_state"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(args.phase == "joint")
    if args.phase == "base":
        for parameter in model.coarse.parameters():
            parameter.requires_grad_(True)
    elif args.phase == "refine":
        for parameter in model.refine.parameters():
            parameter.requires_grad_(True)
    parameter_groups = []
    coarse_parameters = [p for p in model.coarse.parameters() if p.requires_grad]
    refine_parameters = [p for p in model.refine.parameters() if p.requires_grad]
    active_lrs = dict(phase_learning_rates(args.phase, args))
    if coarse_parameters:
        parameter_groups.append({"params": coarse_parameters, "lr": active_lrs["coarse"]})
    if refine_parameters:
        parameter_groups.append({"params": refine_parameters, "lr": active_lrs["refine"]})
    optimizer = torch.optim.Adam(parameter_groups)
    # Preserve the historical 30/80 and 60/80 LR milestones on the matched step budget.
    milestones = [round(args.total_iter * 30 / 80), round(args.total_iter * 60 / 80)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    if args.parallelism == "dp":
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
    elif distributed:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=args.variant.endswith("e1") and args.phase != "refine")

    protocol = {
        "variant": args.variant,
        "phase": args.phase,
        "implementation_revision": 5,
        "data_root": str(Path(args.data_root).resolve()),
        "train_geometry": f"random_aligned_{args.crop_size}_hr_{args.crop_size // 2}_lr_patch",
        "validation_geometry": "full_image",
        "batch_size_per_rank": args.batch_size_per_rank,
        "gradient_accumulation": args.grad_accumulation,
        "global_batch_size": args.batch_size_per_rank * world * args.grad_accumulation,
        "total_iter": args.total_iter,
        "sample_exposures": args.batch_size_per_rank * world * args.grad_accumulation * args.total_iter,
        "parallelism": args.parallelism,
        "world_size": world,
        "backend": dist.get_backend() if distributed else ("data_parallel" if args.parallelism == "dp" else "single"),
        "num_workers": args.num_workers,
        "optimizer": "Adam",
        "lr_base": args.lr_base,
        "lr_refine": args.lr_refine,
        "lr_joint_coarse": args.lr_joint_coarse,
        "lr_joint_refine": args.lr_joint_refine,
        "active_learning_rates": [lr for _, lr in phase_learning_rates(args.phase, args)],
        "scheduler": "MultiStepLR(gamma=0.5)",
        "scheduler_milestones": milestones,
        "seed": args.seed,
        "val_count": args.val_count,
        "loss": ("historical_dynamic_balanced_MSE_Sobel" if args.phase == "base" else
                 "delta_cascade_0.1_0.3_0.6_edge_guided" if args.phase == "refine" else
                 "delta_cascade_0.1_0.3_0.6_edge_guided + 0.5 coarse SmoothL1"),
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()) if args.base_checkpoint else None,
        "refine_checkpoint": str(Path(args.refine_checkpoint).resolve()) if args.refine_checkpoint else None,
        "train_ids": [raw.ids[i] for i in train_ids],
        "validation_ids": [raw.ids[i] for i in val_ids],
    }
    iteration, best_mse = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        if checkpoint.get("implementation_revision") != 5:
            raise ValueError("Cannot resume an obsolete implementation/protocol; use a new matched-exposure directory")
        if checkpoint.get("phase") != args.phase:
            raise ValueError(f"resume checkpoint phase is {checkpoint.get('phase')!r}, expected {args.phase!r}")
        if checkpoint.get("protocol") != protocol:
            raise ValueError("Resume protocol does not match this run")
        (model.module if isinstance(model, (DDP, DataParallel)) else model).load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        iteration = int(checkpoint["iteration"])
        best_mse = float(checkpoint["best_val_mse"])
    if is_main:
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    progress = tqdm(total=args.total_iter - iteration, disable=not is_main,
                    desc=f"{args.variant} {args.phase}", unit="step", dynamic_ncols=True)
    try:
        epoch = 0
        train_iter = iter(train_loader)
        while iteration < args.total_iter:
            model.train()
            if args.phase == "refine":
                (model.module if isinstance(model, (DDP, DataParallel)) else model).coarse.eval()
            optimizer.zero_grad(set_to_none=True)
            train_sum = 0.0
            for _ in range(args.grad_accumulation):
                try:
                    lr, hr, gt = next(train_iter)
                except StopIteration:
                    epoch += 1
                    if sampler is not None:
                        sampler.set_epoch(epoch)
                    train_iter = iter(train_loader)
                    lr, hr, gt = next(train_iter)
                lr, hr, gt = lr.to(device), hr.to(device), gt.to(device)
                outputs = model(lr, hr)
                loss = phase_loss(outputs, gt, args.phase)
                (loss / args.grad_accumulation).backward()
                train_sum += float(loss.detach().item()) / args.grad_accumulation
            optimizer.step()
            scheduler.step()
            iteration += 1
            progress.update(1)

            validate_now = iteration % args.val_every == 0 or iteration == args.total_iter
            save_now = iteration % args.save_every == 0 or validate_now
            if not validate_now and not save_now:
                continue

            improved = False
            if validate_now:
                model.eval()
                val_sum, val_count_actual = 0.0, 0
                with torch.no_grad():
                    for lr, hr, gt in val_loader:
                        lr, hr, gt = lr.to(device), hr.to(device), gt.to(device)
                        outputs = model(lr, hr)
                        outputs["_phase"] = args.phase
                        mse, _ = metric(outputs, gt)
                        val_sum += mse
                        val_count_actual += 1
                val_sum, val_count_actual = reduce_pair(device, val_sum, val_count_actual)
                val_mse = val_sum / max(val_count_actual, 1.0)
                val_psnr = 10.0 * math.log10(1.0 / max(val_mse, 1e-12))
                improved = val_mse < best_mse
                if improved:
                    best_mse = val_mse
                if is_main:
                    record = {"iteration": iteration, "train_loss": train_sum,
                              "val_mse": val_mse, "val_psnr_db": val_psnr,
                              "best_val_mse": best_mse,
                              "lr_values": [group["lr"] for group in optimizer.param_groups]}
                    with log_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record) + "\n")
                    progress.set_postfix(train=f"{train_sum:.4g}", val_psnr=f"{val_psnr:.3f}")
            if is_main:
                save_checkpoint(latest, model, optimizer, scheduler, iteration, best_mse, protocol)
                if improved:
                    save_checkpoint(best, model, optimizer, scheduler, iteration, best_mse, protocol)
        if is_main:
            save_checkpoint(final, model, optimizer, scheduler, iteration, best_mse, protocol)
    finally:
        if distributed and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    main()
