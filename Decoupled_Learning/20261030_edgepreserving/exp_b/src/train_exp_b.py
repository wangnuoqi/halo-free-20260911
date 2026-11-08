"""Train Exp B in one of the base, refine, or joint phases."""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

try:
    from exp_b.src.data_exp_b import (
        PairedPatchCollator,
        PairedTripletDataset,
        StridedEvalSampler,
        deterministic_train_val_indices,
        seed_worker,
    )
    from exp_b.src.model_exp_b import FullExpB, sobel
except ImportError:  # direct execution as python exp_b/src/train_exp_b.py
    from data_exp_b import (
        PairedPatchCollator,
        PairedTripletDataset,
        StridedEvalSampler,
        deterministic_train_val_indices,
        seed_worker,
    )
    from model_exp_b import FullExpB, sobel


RESUME_PROTOCOL_KEYS = (
    "architecture", "validation_geometry", "base_checkpoint_sha256", "refine_checkpoint_sha256",
    "phase", "data_root", "crop_size", "batch_size_per_rank", "world_size",
    "global_batch_size", "precision", "lr", "min_lr", "lr_schedule",
    "weight_decay", "seed", "val_count", "base_gradient_weight", "epochs",
)
INITIALIZATION_PROTOCOL_KEYS = (
    "architecture", "data_root", "crop_size", "batch_size_per_rank",
    "world_size", "global_batch_size", "precision", "seed", "val_count",
    "epochs", "lr", "min_lr", "lr_schedule", "weight_decay",
)


def compute_sobel_grad(image: torch.Tensor, eps: float = 1e-6, normalize: bool = False):
    magnitude, gx, gy = sobel(image.float(), eps=eps)
    if normalize:
        magnitude = magnitude / (magnitude.amax(dim=(2, 3), keepdim=True) + eps)
    return magnitude, gx, gy


def batched_otsu_threshold(
    gradient: torch.Tensor, num_bins: int = 256, eps: float = 1e-6
) -> torch.Tensor:
    if gradient.ndim != 4 or gradient.shape[1] != 1:
        raise ValueError("gradient must have shape (B,1,H,W)")
    batch = gradient.shape[0]
    flat = gradient.float().flatten(1)
    low = flat.min(dim=1).values
    high = flat.max(dim=1).values
    span_raw = high - low
    span = span_raw.clamp_min(eps)
    normalized = ((flat - low[:, None]) / span[:, None]).clamp(0.0, 1.0)
    bin_index = (normalized * (num_bins - 1)).long()
    offsets = torch.arange(batch, device=gradient.device)[:, None] * num_bins
    hist = torch.bincount(
        (bin_index + offsets).reshape(-1), minlength=batch * num_bins
    ).reshape(batch, num_bins).float()
    probability = hist / hist.sum(dim=1, keepdim=True).clamp_min(eps)
    centers = torch.linspace(0.0, 1.0, num_bins, device=gradient.device)[None, :]
    omega = probability.cumsum(dim=1)
    mean = (probability * centers).cumsum(dim=1)
    total_mean = mean[:, -1:]
    between = ((total_mean * omega - mean) ** 2) / (
        omega * (1.0 - omega)
    ).clamp_min(eps)
    between[:, 0] = 0.0
    between[:, -1] = 0.0
    threshold = low + centers[0, between.argmax(dim=1)] * span
    return torch.where(span_raw > eps, torch.minimum(threshold, high - eps), high)


def edge_guided_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """exp7C edge/smooth pixel loss plus edge-only Sobel-vector loss."""
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    prediction = prediction.float()
    target = target.float()
    gradient, _, _ = compute_sobel_grad(target, normalize=True)
    threshold = batched_otsu_threshold(gradient)
    edge = (gradient > threshold.view(-1, 1, 1, 1)).float()
    empty = edge.flatten(1).sum(dim=1) <= 0
    if empty.any():
        flattened = edge.flatten(1)
        fallback = torch.zeros_like(flattened)
        fallback.scatter_(1, gradient.flatten(1).argmax(dim=1, keepdim=True), 1.0)
        flattened[empty] = fallback[empty]
        edge = flattened.view_as(edge)
    edge = edge.detach()
    smooth = (1.0 - edge).clamp(0.0, 1.0)
    pixel = (prediction - target).abs()
    _, pred_gx, pred_gy = compute_sobel_grad(prediction, normalize=True)
    _, target_gx, target_gy = compute_sobel_grad(target, normalize=True)
    gradient_error = (pred_gx - target_gx).abs() + (pred_gy - target_gy).abs()
    edge_pixel = (pixel * edge).sum() / edge.sum().clamp_min(1.0)
    smooth_pixel = (pixel * smooth).sum() / smooth.sum().clamp_min(1.0)
    edge_gradient = (gradient_error * edge).sum() / edge.sum().clamp_min(1.0)
    return smooth_pixel + 2.0 * edge_pixel + 0.2 * edge_gradient


def base_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gradient_weight: float = 0.1,
) -> torch.Tensor:
    """Base objective: image MSE plus a small Sobel-gradient L1 term."""
    prediction = prediction.float()
    target = target.float()
    _, pred_gx, pred_gy = compute_sobel_grad(prediction)
    _, target_gx, target_gy = compute_sobel_grad(target)
    gradient_l1 = 0.5 * (
        F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)
    )
    return F.mse_loss(prediction, target) + gradient_weight * gradient_l1


def delta_cascade_loss(
    outputs: dict[str, torch.Tensor], target: torch.Tensor
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    coarse0 = outputs["coarse0"].float()
    delta_gt = target.float() - coarse0
    l1 = edge_guided_loss(outputs["delta1"], delta_gt)
    l2 = edge_guided_loss(outputs["delta2"], delta_gt)
    l3 = edge_guided_loss(outputs["delta3"], delta_gt)
    return 0.1 * l1 + 0.3 * l2 + 0.6 * l3, (l1, l2, l3)


def legacy_cascade_loss(
    outputs: dict[str, torch.Tensor], target: torch.Tensor
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Delta-space cascade objective plus the historical coarse fidelity term."""
    cascade, parts = delta_cascade_loss(outputs, target)
    coarse = F.smooth_l1_loss(outputs["coarse0"].float(), target.float())
    return cascade + 0.5 * coarse, parts


def init_distributed(parallelism: str, requested_device: str):
    if parallelism == "single":
        if requested_device == "cuda" or (
            requested_device == "auto" and torch.cuda.is_available()
        ):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
            return False, 0, 1, torch.device("cuda", 0), None
        return False, 0, 1, torch.device("cpu"), None
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("DDP launch requires torchrun and LOCAL_RANK")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    visible = torch.cuda.device_count()
    if visible < world_size:
        raise RuntimeError(f"torchrun requested {world_size} ranks, but {visible} CUDA devices are visible")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return True, rank, world_size, torch.device("cuda", local_rank), "nccl"


def seed_everything(seed: int, rank: int, device: torch.device) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.random.default_generator.manual_seed(value)
    if device.type == "cuda":
        torch.cuda.manual_seed(value)


def unwrap(module):
    return module.module if isinstance(module, DDP) else module


def destroy_process_group_safely() -> None:
    """Try to release DDP resources without hiding a training exception.

    NCCL teardown can itself fail after a CUDA/NCCL error.  Since this helper
    is called from ``finally``, let that failure be diagnostic only; otherwise
    it would replace the exception that caused training to stop (or turn a
    successfully saved run into a reported training failure).
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    try:
        dist.destroy_process_group()
    except Exception as exc:  # cleanup must not mask the original outcome
        try:
            rank = dist.get_rank()
        except Exception:
            rank = "unknown"
        print(
            f"Warning: rank {rank} failed to destroy the distributed process group: {exc}",
            flush=True,
        )


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"not an Exp B checkpoint: {path}")
    return checkpoint


def load_full_state(model: FullExpB, checkpoint: dict) -> None:
    state = checkpoint["model_state"]
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def validate_initialization_protocol(saved: dict, current: dict, source: Path) -> None:
    differences = {
        key: (saved.get(key), current.get(key))
        for key in INITIALIZATION_PROTOCOL_KEYS
        if saved.get(key) != current.get(key)
    }
    if differences:
        raise ValueError(f"initial checkpoint protocol differs for {source}: {differences}")


def initialize_phase(model: FullExpB, args, protocol: dict) -> None:
    if getattr(args, "expe_variant", None):
        from expe.refine_joint import initialize_phase as initialize_expe
        return initialize_expe(model, args, protocol)
    if args.phase == "base":
        return
    if not args.base_checkpoint:
        raise ValueError(f"--base-checkpoint is required for phase={args.phase}")
    base_ckpt = load_checkpoint(args.base_checkpoint)
    if base_ckpt.get("phase") != "base":
        raise ValueError(f"expected a base checkpoint, got phase={base_ckpt.get('phase')!r}")
    validate_initialization_protocol(
        base_ckpt.get("protocol", {}), protocol, Path(args.base_checkpoint)
    )
    load_full_state(model, base_ckpt)
    if args.phase == "refine":
        return
    if not args.refine_checkpoint:
        raise ValueError("--refine-checkpoint is required for phase=joint")
    refine_ckpt = load_checkpoint(args.refine_checkpoint)
    if refine_ckpt.get("phase") != "refine":
        raise ValueError(f"expected a refine checkpoint, got phase={refine_ckpt.get('phase')!r}")
    validate_initialization_protocol(
        refine_ckpt.get("protocol", {}), protocol, Path(args.refine_checkpoint)
    )
    original_base = refine_ckpt.get("protocol", {}).get("base_checkpoint")
    supplied_base = str(Path(args.base_checkpoint).resolve())
    if original_base and str(Path(original_base).resolve()) != supplied_base:
        raise ValueError(
            "refine checkpoint was trained from a different base checkpoint: "
            f"{original_base} != {supplied_base}"
        )
    refine_state = refine_ckpt["model_state"]
    prefix = "refine."
    selected = {key[len(prefix):]: value for key, value in refine_state.items() if key.startswith(prefix)}
    if not selected:
        raise ValueError("refine checkpoint does not contain the FullExpB refine module")
    model.refine.load_state_dict(selected, strict=True)


def set_trainable_phase(model: FullExpB, phase: str) -> torch.nn.Module:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if phase == "base":
        selected = model.base
    elif phase == "refine":
        selected = model.refine
    else:
        selected = model
    for parameter in selected.parameters():
        parameter.requires_grad_(True)
    return selected


def forward_phase(
    phase: str,
    full_model: FullExpB,
    train_module: torch.nn.Module,
    lr: torch.Tensor,
    hr: torch.Tensor,
):
    lr_centered = lr - 0.5
    hr_centered = hr - 0.5
    if phase == "base":
        coarse = train_module(lr_centered, hr_centered)
        return {"coarse0": coarse}
    if phase == "refine":
        with torch.no_grad():
            coarse = full_model.base(lr_centered, hr_centered)
        refined = train_module(hr_centered, coarse - 0.5)
        outputs = {"coarse0": coarse, **refined}
        for key in ("coarse1", "coarse2", "final"):
            outputs[key] = (outputs[key] + 0.5).clamp(0.0, 1.0)
        return outputs
    return train_module(lr_centered, hr_centered)


def set_phase_mode(full_model: FullExpB, phase: str, train_module: torch.nn.Module) -> None:
    full_model.eval()
    if phase == "base":
        full_model.base.train()
    elif phase == "refine":
        full_model.refine.train()
    else:
        full_model.train()
    train_module.train()


def selected_prediction(phase: str, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return outputs["coarse0"] if phase == "base" else outputs["final"]


def make_protocol(args, world_size: int) -> dict:
    record = {
        "phase": args.phase,
        "data_root": str(Path(args.data_root).resolve()),
        "crop_size": args.crop_size,
        "batch_size_per_rank": args.batch_size_per_rank,
        "world_size": world_size,
        "global_batch_size": args.batch_size_per_rank * world_size,
        "precision": args.precision,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "lr_schedule": args.lr_schedule,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "val_count": args.val_count,
        "base_gradient_weight": args.base_gradient_weight,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "architecture": "exp_b_original_lrnet_macro_viv_blocks_v1",
        "train_geometry": "random_aligned_128_hr_64_lr_patch",
        "inference_geometry": "full_image_even_border_crop",
        "pin_memory": bool(args.pin_memory),
        "init_source": (
            "random"
            if args.phase == "base"
            else (
                "base_best"
                if args.phase == "refine"
                else "base_best_plus_refine_best"
            )
        ),
        "base_checkpoint": (
            str(Path(args.base_checkpoint).resolve()) if args.base_checkpoint else None
        ),
        "refine_checkpoint": (
            str(Path(args.refine_checkpoint).resolve()) if args.refine_checkpoint else None
        ),
    }

    if getattr(args, "expe_variant", None):
        from expe.refine_joint import extend_protocol
        extend_protocol(record, args)
    return record


def dataloader_runtime_options(num_workers: int, pin_memory: bool) -> dict:
    return {
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
        "persistent_workers": num_workers > 0,
    }


def validate_resume_protocol(saved: dict, current: dict) -> None:
    differences = {
        key: (saved.get(key), current.get(key))
        for key in RESUME_PROTOCOL_KEYS
        if saved.get(key) != current.get(key)
    }
    if differences:
        raise ValueError(f"resume protocol differs in immutable fields: {differences}")


def reduce_metrics(values: torch.Tensor, distributed: bool) -> torch.Tensor:
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


@torch.no_grad()
def validate(
    phase: str,
    full_model: FullExpB,
    train_module: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    distributed: bool,
) -> dict[str, float]:
    full_model.eval()
    train_module.eval()
    stats = torch.zeros(3, device=device, dtype=torch.float64)
    for lr, hr, gt in loader:
        lr = lr.to(device=device, dtype=torch.float32)
        hr = hr.to(device=device, dtype=torch.float32)
        gt = gt.to(device=device, dtype=torch.float32)
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if precision == "bf16" and device.type == "cuda"
            else contextlib.nullcontext()
        )
        with amp:
            outputs = forward_phase(phase, full_model, train_module, lr, hr)
        prediction = selected_prediction(phase, outputs).float().clamp(0.0, 1.0)
        mse = (prediction - gt).square().mean()
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
        stats[0] += mse.double()
        stats[1] += psnr.double()
        stats[2] += 1.0
    reduce_metrics(stats, distributed)
    count = max(1.0, float(stats[2].item()))
    return {"mean_image_mse": float((stats[0] / count).item()),
            "mean_image_psnr_db": float((stats[1] / count).item()),
            "images": int(stats[2].item())}


def save_checkpoint(
    path: Path,
    model: FullExpB,
    optimizer,
    scheduler,
    phase: str,
    epoch: int,
    step: int,
    best_mse: float,
    protocol: dict,
    val_metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "phase": phase,
            "epoch": epoch,
            "step": step,
            "best_val_mse": best_mse,
            "val_metrics": val_metrics,
            "protocol": protocol,
        },
        path,
    )


def write_history(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def needs_unused_parameter_detection(phase: str, expe_variant) -> bool:
    # Refine wraps only ThreeStageRefine; Exp E joint also wraps VIVNet,
    # whose upstream reduce_chan_level2 weight is absent from the loss graph.
    return bool(expe_variant) and expe_variant != "exp7c_e0" and phase == "joint"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("base", "refine", "joint"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--refine-checkpoint", default=None)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--batch-size-per-rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-6)
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-gradient-weight", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--parallelism", choices=("ddp", "single"), default="ddp")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expe-variant", default=None, help="Exp E coarse architecture, e.g. viv_gf_e1")
    parser.add_argument("--full-image-validation", action="store_true")
    args = parser.parse_args()
    if args.expe_variant and args.phase == "base":
        parser.error("Train Exp E Base using expc/src/train_expc.py")
    if args.crop_size <= 0 or args.batch_size_per_rank <= 0 or args.epochs <= 0:
        parser.error("crop-size, batch-size-per-rank, and epochs must be positive")
    if args.num_workers < 0 or args.val_count <= 0:
        parser.error("num-workers must be nonnegative and val-count positive")
    if args.max_steps < 0:
        parser.error("max-steps must be nonnegative")

    distributed, rank, world_size, device, backend = init_distributed(
        args.parallelism, args.device
    )
    is_main = rank == 0
    seed_everything(args.seed, rank, device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = args.phase
    latest_path = output / f"{prefix}_latest.pth"
    best_path = output / f"{prefix}_best.pth"
    final_path = output / f"{prefix}_final.pth"
    history_path = output / "loss_history.jsonl"
    protocol = make_protocol(args, world_size)

    if not args.resume and any(path.exists() for path in (latest_path, best_path, final_path, history_path)):
        raise FileExistsError(f"Existing run in {output}; use --resume or a new output directory")
    if args.expe_variant:
        from expe.refine_joint import FullExpE
        model = FullExpE(args.expe_variant)
    else:
        model = FullExpB()
    resume_checkpoint = None
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"--resume requested, but latest checkpoint is absent: {latest_path}")
        resume_checkpoint = load_checkpoint(latest_path)
        if resume_checkpoint.get("phase") != args.phase:
            raise ValueError(f"latest checkpoint phase is {resume_checkpoint.get('phase')!r}, expected {args.phase!r}")
        load_full_state(model, resume_checkpoint)
    else:
        initialize_phase(model, args, protocol)

    trainable = set_trainable_phase(model, args.phase)
    model.to(device)
    trainable = {
        "base": model.base,
        "refine": model.refine,
        "joint": model,
    }[args.phase]
    if distributed:
        trainable = DDP(trainable, device_ids=[device.index], find_unused_parameters=needs_unused_parameter_detection(args.phase, args.expe_variant))

    optimizer = torch.optim.AdamW(
        [parameter for parameter in trainable.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr
        )
        if args.lr_schedule == "cosine"
        else None
    )
    if resume_checkpoint is not None:
        validate_resume_protocol(resume_checkpoint.get("protocol", {}), protocol)
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if scheduler is not None and resume_checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        start_epoch = int(resume_checkpoint["epoch"])
        step = int(resume_checkpoint["step"])
        best_mse = float(resume_checkpoint.get("best_val_mse", math.inf))
    else:
        start_epoch, step, best_mse = 0, 0, math.inf
    if is_main:
        (output / "config.json").write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    raw_train = PairedTripletDataset(args.data_root, split="train")
    split_ids = raw_train.ids
    if args.expe_variant:
        from expe.refine_joint import canonical_ids
        split_ids = canonical_ids(split_ids)
    train_indices, val_indices = deterministic_train_val_indices(
        split_ids, val_count=args.val_count, seed=args.seed
    )
    if args.expe_variant:
        from expe.refine_joint import check_split
        check_split(args, raw_train.ids, train_indices, val_indices, protocol)
        if is_main:
            (output / "config.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    train_dataset = Subset(raw_train, train_indices)
    val_dataset = Subset(raw_train, val_indices)
    train_sampler = (
        DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank,
            shuffle=True, seed=args.seed, drop_last=True,
        )
        if distributed
        else None
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed + rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_rank,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **dataloader_runtime_options(args.num_workers, args.pin_memory),
        drop_last=distributed,
        collate_fn=PairedPatchCollator(args.crop_size, random_crop=True),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_sampler = StridedEvalSampler(val_dataset, rank, world_size) if distributed else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        **dataloader_runtime_options(args.num_workers, args.pin_memory),
        collate_fn=PairedPatchCollator(crop_size=0 if args.full_image_validation else args.crop_size, random_crop=False),
        worker_init_fn=seed_worker,
    )
    if is_main:
        print(
            f"{args.expe_variant or 'Exp B'} {args.phase}: backend={backend or 'single'}, train={len(train_indices)}, "
            f"val={len(val_indices)}, world={world_size}, batch/rank={args.batch_size_per_rank}, "
            f"global_batch={args.batch_size_per_rank * world_size}, crop={args.crop_size}, "
            f"workers/rank={args.num_workers}, pin_memory={args.pin_memory}, precision={args.precision}",
            flush=True,
        )

    start_time = time.time()
    completed_epoch = start_epoch
    val_metrics = (
        resume_checkpoint.get("val_metrics", {
            "mean_image_mse": best_mse if math.isfinite(best_mse) else math.inf,
            "mean_image_psnr_db": 0.0,
            "images": 0,
        })
        if resume_checkpoint is not None
        else {"mean_image_mse": math.inf, "mean_image_psnr_db": 0.0, "images": 0}
    )
    progress = tqdm(
        total=args.epochs,
        initial=start_epoch,
        desc=f"{args.expe_variant or 'Exp B'} {args.phase}",
        unit="epoch",
        dynamic_ncols=True,
        disable=not is_main,
    )
    try:
        first_epoch = (
            args.epochs
            if args.max_steps and step >= args.max_steps
            else start_epoch
        )
        for epoch in range(first_epoch, args.epochs):
            set_phase_mode(model, args.phase, trainable)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            loss_sum = torch.zeros(2, device=device, dtype=torch.float64)
            for batch_index, (lr, hr, gt) in enumerate(train_loader):
                lr = lr.to(device=device, dtype=torch.float32)
                hr = hr.to(device=device, dtype=torch.float32)
                gt = gt.to(device=device, dtype=torch.float32)
                autocast = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if args.precision == "bf16" and device.type == "cuda"
                    else contextlib.nullcontext()
                )
                with autocast:
                    outputs = forward_phase(args.phase, model, trainable, lr, hr)
                if args.phase == "base":
                    loss = base_loss(
                        outputs["coarse0"], gt, gradient_weight=args.base_gradient_weight
                    )
                else:
                    cascade, _ = delta_cascade_loss(outputs, gt)
                    loss = cascade
                    if args.phase == "joint":
                        loss = loss + 0.5 * F.smooth_l1_loss(
                            outputs["coarse0"].float(), gt
                        )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                step += 1
                loss_sum[0] += loss.detach().double()
                loss_sum[1] += 1.0
                if is_main and (step == 1 or step % 50 == 0):
                    write_history(output / "train_steps.jsonl", {"phase": args.phase, "epoch": epoch + 1, "step": step, "loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
                if args.max_steps and step >= args.max_steps:
                    break
            reduce_metrics(loss_sum, distributed)
            train_mean = float((loss_sum[0] / loss_sum[1].clamp_min(1)).item())
            if scheduler is not None:
                scheduler.step()
            completed_epoch = epoch + 1
            val_metrics = validate(
                args.phase, model, unwrap(trainable), val_loader, device,
                args.precision, distributed,
            )
            improved = val_metrics["mean_image_mse"] < best_mse
            if improved:
                best_mse = val_metrics["mean_image_mse"]
            row = {
                "phase": args.phase,
                "epoch": completed_epoch,
                "step": step,
                "train_loss": train_mean,
                "val_mean_image_mse": val_metrics["mean_image_mse"],
                "val_mean_image_psnr_db": val_metrics["mean_image_psnr_db"],
                "best_val_mse": best_mse,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - start_time,
            }
            if is_main:
                write_history(history_path, row)
                save_checkpoint(
                    latest_path, model, optimizer, scheduler, args.phase, completed_epoch,
                    step, best_mse, protocol, val_metrics,
                )
                if improved:
                    save_checkpoint(
                        best_path, model, optimizer, scheduler, args.phase, completed_epoch,
                        step, best_mse, protocol, val_metrics,
                    )
                progress.set_postfix(
                    loss=f"{train_mean:.3g}",
                    val_mse=f"{val_metrics['mean_image_mse']:.3g}",
                    psnr=f"{val_metrics['mean_image_psnr_db']:.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2g}",
                )
                progress.update(1)
            if distributed:
                dist.barrier()
            if args.max_steps and step >= args.max_steps:
                break

        if is_main:
            save_checkpoint(
                final_path, model, optimizer, scheduler, args.phase, completed_epoch,
                step, best_mse, protocol, val_metrics,
            )
        if distributed:
            dist.barrier()
    finally:
        progress.close()
        if distributed:
            destroy_process_group_safely()


if __name__ == "__main__":
    main()
