"""Four-GPU DDP training for the four Exp C single-image WLS models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import chain

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
BASELINE = PROJECT.parent / "20260815_edgepreserving" / "baseline"
sys.path.insert(0, str(BASELINE.parent))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "exp_a" / "src"))
sys.path.insert(0, str(BASELINE))

from baseline.wls_data import WLSPairs, collate_pairs, partition_ids
from models import GUIDED_VARIANTS, VARIANTS, build_model
from expe.checkpoint import initialize_lr, initialize_model
from expe.losses import (LOSS_MODES, ROUTE_SUPERVISION_WEIGHT,
                         experiment_loss, route_supervision_loss)
from expe.model import EXPE_VARIANTS


def ddp_barrier() -> None:
    """Synchronize with the rank's explicitly bound CUDA device."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


class RandomPatchPairs(Dataset):
    """Read paired float32 images from a read-only memory map and return one random crop."""

    def __init__(self, cache_path, indices, crop_size):
        self.cache_path = str(cache_path)
        self.indices = indices
        self.crop_size = crop_size
        self._cache = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self._cache is None:
            self._cache = np.load(
                self.cache_path, mmap_mode="r", allow_pickle=False
            )
        pair = self._cache[self.indices[index]]
        height, width = pair.shape[-2:]
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(
                f"Cached image {height}x{width} is smaller than crop {self.crop_size}"
            )
        top = int(torch.randint(height - self.crop_size + 1, ()).item())
        left = int(torch.randint(width - self.crop_size + 1, ()).item())
        image = np.array(pair[0, top : top + self.crop_size, left : left + self.crop_size], copy=True)
        target = np.array(pair[1, top : top + self.crop_size, left : left + self.crop_size], copy=True)
        return torch.from_numpy(image[None]), torch.from_numpy(target[None])


class RandomGuidedPatchPairs(RandomPatchPairs):
    """Aligned LR/HR/target crops backed by the shared float32 memory maps."""
    def __init__(self, cache_path, lr_cache_path, indices, crop_size):
        super().__init__(cache_path, indices, crop_size)
        self.lr_cache_path = str(lr_cache_path)
        self._lr_cache = None

    def __getitem__(self, index):
        if self._cache is None:
            self._cache = np.load(self.cache_path, mmap_mode="r", allow_pickle=False)
        if self._lr_cache is None:
            self._lr_cache = np.load(self.lr_cache_path, mmap_mode="r", allow_pickle=False)
        image_index = self.indices[index]
        height, width = self._cache.shape[-2:]
        crop = self.crop_size
        if crop % 2:
            raise ValueError("guided-filter crop size must be divisible by its 2x scale")
        top = int(torch.randint((height - crop) // 2 + 1, ()).item()) * 2
        left = int(torch.randint((width - crop) // 2 + 1, ()).item()) * 2
        half = crop // 2
        image = np.array(self._cache[image_index, 0, top:top + crop, left:left + crop], copy=True)
        target = np.array(self._cache[image_index, 1, top:top + crop, left:left + crop], copy=True)
        lr = np.array(self._lr_cache[image_index, 0, top // 2:top // 2 + half,
                                     left // 2:left // 2 + half], copy=True)
        return torch.from_numpy(lr[None]), torch.from_numpy(image[None]), torch.from_numpy(target[None])


class GuidedTriplets(Dataset):
    """Read the exact precomputed half-resolution input paired with HR and WLS target."""
    def __init__(self, dataset, data_root, split):
        self.dataset = dataset
        base = dataset.dataset if isinstance(dataset, Subset) else dataset
        indices = dataset.indices if isinstance(dataset, Subset) else range(len(dataset))
        self.ids = [base.ids[index] for index in indices]
        suffix = "" if split == "train" else "-200"
        directory = Path(data_root) / f"{split}input-l0p500-a1p400{suffix}_downsample"
        paths = {path.stem.removesuffix("-log"): path for path in directory.iterdir() if path.is_file()}
        missing = set(self.ids) - set(paths)
        if missing:
            raise ValueError(f"missing paired LR inputs: {sorted(missing)[:5]}")
        self.lr_paths = [paths[identifier] for identifier in self.ids]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        lr = cv2.imread(str(self.lr_paths[index]), cv2.IMREAD_UNCHANGED)
        if lr is None or lr.ndim != 2:
            raise ValueError(f"invalid LR input: {self.lr_paths[index]}")
        lr = torch.from_numpy(np.ascontiguousarray(lr.astype(np.float32)[None]))
        image, target = self.dataset[index]
        if image.shape[-2:] != (lr.shape[-2] * 2, lr.shape[-1] * 2):
            raise ValueError(f"expected 2x LR/HR geometry for {self.ids[index]}")
        return lr, image, target


def collate_guided(batch):
    height = min(item[1].shape[-2] for item in batch)
    width = min(item[1].shape[-1] for item in batch)
    height -= height % 2
    width -= width % 2
    return (
        torch.stack([lr[..., :height // 2, :width // 2] for lr, _, _ in batch]),
        torch.stack([image[..., :height, :width] for _, image, _ in batch]),
        torch.stack([target[..., :height, :width] for _, _, target in batch]),
    )


def ensure_float32_cache(dataset, cache_path, rank, parallelism):
    """Decode each TIFF pair once into a reusable float32 NPY memory map."""
    cache_path = Path(cache_path)
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    ids_digest = hashlib.sha256("\n".join(dataset.ids).encode()).hexdigest()
    expected = {"ids_sha256": ids_digest, "count": len(dataset), "dtype": "float32"}

    if rank == 0:
        valid = False
        if cache_path.exists() and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                mapped = np.load(cache_path, mmap_mode="r")
                valid = (
                    all(metadata.get(key) == value for key, value in expected.items())
                    and mapped.ndim == 4 and mapped.shape[:2] == (len(dataset), 2)
                    and mapped.dtype == np.float32
                )
                if not valid:
                    print(f"float32 cache metadata/shape mismatch: metadata={metadata}, "
                          f"array_shape={mapped.shape}, dtype={mapped.dtype}", flush=True)
                del mapped
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"float32 cache validation failed for {cache_path}: {exc!r}", flush=True)
                valid = False
        else:
            print(f"float32 cache missing: data={cache_path.exists()}, "
                  f"metadata={metadata_path.exists()}, path={cache_path}", flush=True)
        if not valid:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            first_image, first_target = dataset[0]
            if first_image.shape != first_target.shape:
                raise ValueError("WLS input and target shapes differ")
            height, width = first_image.shape[-2:]
            mapped = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.float32,
                shape=(len(dataset), 2, height, width),
            )
            cv2.setNumThreads(1)
            with ThreadPoolExecutor(max_workers=8) as pool:
                pairs = pool.map(dataset.__getitem__, range(1, len(dataset)))
                for index, (image, target) in enumerate(
                    chain(((first_image, first_target),), pairs)
                ):
                    if image.shape != (1, height, width) or target.shape != image.shape:
                        raise ValueError(f"Inconsistent WLS image shape at item {index}")
                    mapped[index, 0] = image[0].numpy()
                    mapped[index, 1] = target[0].numpy()
                    if (index + 1) % 100 == 0 or index + 1 == len(dataset):
                        print(f"float32 TIFF cache: {index + 1}/{len(dataset)} pairs", flush=True)
            mapped.flush()
            del mapped
            os.replace(temporary, cache_path)
            metadata = {**expected, "shape": [len(dataset), 2, height, width]}
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if parallelism == "ddp":
        ddp_barrier()
    return cache_path


def ensure_lr_cache(dataset, data_root, cache_path, rank, parallelism):
    cache_path = Path(cache_path)
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    digest = hashlib.sha256("\n".join(dataset.ids).encode()).hexdigest()
    expected = {"ids_sha256": digest, "count": len(dataset), "dtype": "float32", "scale": 2}
    if rank == 0:
        valid = False
        if cache_path.exists() and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                mapped = np.load(cache_path, mmap_mode="r", allow_pickle=False)
                valid = (all(metadata.get(key) == value for key, value in expected.items())
                         and mapped.ndim == 4 and mapped.shape[:2] == (len(dataset), 1)
                         and mapped.dtype == np.float32)
                del mapped
            except (OSError, ValueError, json.JSONDecodeError):
                valid = False
        if not valid:
            folder = Path(data_root) / "traininput-l0p500-a1p400_downsample"
            lr_paths = {path.stem.removesuffix("-log"): path for path in folder.iterdir() if path.is_file()}
            first = cv2.imread(str(lr_paths[dataset.ids[0]]), cv2.IMREAD_UNCHANGED)
            if first is None or first.ndim != 2:
                raise ValueError("could not read first half-resolution LR image")
            shape = (len(dataset), 1, first.shape[0], first.shape[1])
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            mapped = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
            for index, identifier in enumerate(dataset.ids):
                path = lr_paths.get(identifier)
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path else None
                if image is None or image.shape != shape[-2:]:
                    raise ValueError(f"invalid LR cache source for {identifier}: {path}")
                mapped[index, 0] = image.astype(np.float32)
                if (index + 1) % 200 == 0 or index + 1 == len(dataset):
                    print(f"LR float32 TIFF cache: {index + 1}/{len(dataset)}", flush=True)
            mapped.flush()
            del mapped
            os.replace(temporary, cache_path)
            metadata_path.write_text(json.dumps({**expected, "shape": shape}, indent=2) + "\n",
                                     encoding="utf-8")
    if parallelism == "ddp":
        ddp_barrier()
    return cache_path


def seed_all(seed: int, rank: int, parallelism: str = "ddp") -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.random.default_generator.manual_seed(value)
    if parallelism == "dp":
        torch.cuda.manual_seed_all(value)
    else:
        torch.cuda.manual_seed(value)


def even(x):
    return x[..., :x.shape[-2] - x.shape[-2] % 2,
             :x.shape[-1] - x.shape[-1] % 2]


def edge_loss(prediction, target):
    pixel = torch.sqrt((prediction - target).square() + 1e-6).mean()
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    true_dx = target[..., :, 1:] - target[..., :, :-1]
    true_dy = target[..., 1:, :] - target[..., :-1, :]
    gradient = (pred_dx - true_dx).abs().mean() + (pred_dy - true_dy).abs().mean()
    return pixel + 0.05 * gradient, pixel.detach(), gradient.detach()


def protocol(args, train_ids, val_ids):
    if args.parallelism == "ddp":
        world = int(os.environ.get("WORLD_SIZE", "1"))
    elif args.parallelism == "dp":
        world = torch.cuda.device_count()
    else:
        world = 1
    legacy_e0 = args.variant == "exp7c_e0"
    record = {
        "variant": args.variant,
        "architecture": ("LegacyExp7C" if legacy_e0 else
                         getattr(VARIANTS[args.variant], "__name__", "ExpEAffine")),
        "task": "single_channel_WLS_l0p500_a1p400",
        "data_root": str(args.data_root.resolve()),
        "train_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "validation_ids_sha256": hashlib.sha256("\n".join(val_ids).encode()).hexdigest(),
        "train_pairs": len(train_ids), "validation_pairs": len(val_ids),
        "train_crop_size": args.crop_size, "validation_geometry": "full_image_even",
        "world_size": world,
        "batch_size_per_gpu": args.batch_size_per_gpu,
        "effective_batch_size": args.batch_size_per_gpu * world * args.grad_accumulation,
        "gradient_accumulation": args.grad_accumulation,
        "parallelism": args.parallelism,
        "backend": (os.environ.get("TORCH_DDP_BACKEND", "nccl").lower()
                    if args.parallelism == "ddp" else (
                        "data_parallel" if args.parallelism == "dp" else "single_gpu"
                    )),
        "precision": args.precision, "optimizer": "AdamW", "lr": args.lr,
        "weight_decay": args.weight_decay, "total_iter": args.total_iter,
        "loss": ("MSE + dynamically_balanced_Sobel_vector_L1_scale_min_0p1"
                 if args.variant in EXPE_VARIANTS and args.loss_mode == "base" else
                 "edge_weighted_Charbonnier + 0.05 * edge_weighted_finite_difference_L1"
                 if args.variant in EXPE_VARIANTS else
                 "Charbonnier + 0.05 * finite_difference_L1"),
        "lr_source": "paired_precomputed_2x_tiff" if args.variant in GUIDED_VARIANTS else None,
        "lr_predictor_depth": (7 if legacy_e0 else 6 if args.variant in EXPE_VARIANTS else
                               4 if args.variant in {"deep_guided_lr4", "guided_pyramid_lr4"} else
                               8 if args.variant == "deep_guided_lr8" else None),
        "lr_predictor_width": (24 if legacy_e0 else 32 if args.variant in EXPE_VARIANTS else
                               24 if args.variant in GUIDED_VARIANTS else None),
        "lr_predictor_refinement_blocks": (0 if args.variant in EXPE_VARIANTS else None),
        "lr_predictor_topology": ("four_dilated_blocks" if legacy_e0 else
                                  "three_scale_viv_unet" if args.variant in EXPE_VARIANTS else
                                  "two_level_pyramid" if args.variant == "guided_pyramid_lr4"
                                  else "flat" if args.variant in GUIDED_VARIANTS else None),
        "guided_filter": ("legacy_conv_guided_filter" if legacy_e0 else
                          args.variant if args.variant in EXPE_VARIANTS else
                          "learned_local_affine_box3_bilinear" if args.variant in GUIDED_VARIANTS else None),
        "seed": args.seed,
    }
    if args.variant in EXPE_VARIANTS:
        def checkpoint_record(path):
            if path is None:
                return None, None
            if not path.is_file():
                raise FileNotFoundError(path)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return str(path.resolve()), digest.hexdigest()

        lr_path, lr_sha = checkpoint_record(args.init_lr_checkpoint)
        model_path, model_sha = checkpoint_record(args.init_model_checkpoint)
        record.update({
            "loss_mode": args.loss_mode,
            "init_lr_checkpoint": lr_path,
            "init_lr_checkpoint_sha256": lr_sha,
            "init_model_checkpoint": model_path,
            "init_model_checkpoint_sha256": model_sha,
            "freeze_lr": args.freeze_lr,
            "free_b": bool(EXPE_VARIANTS[args.variant].get("free_b", False)),
            "branch_edges": bool(EXPE_VARIANTS[args.variant].get("branch_edges", False)),
            "route_supervision": bool(EXPE_VARIANTS[args.variant].get("route_supervision", False)),
            "route_supervision_weight": (ROUTE_SUPERVISION_WEIGHT
                                          if EXPE_VARIANTS[args.variant].get("route_supervision", False)
                                          else 0.0),
        })
    return record


@torch.no_grad()
def validate(model, loader, device, precision):
    model.eval()
    mse_sum = mae_sum = edge_sum = 0.0
    count = 0
    for batch in loader:
        if len(batch) == 3:
            lr, image, target = batch
            lr = lr.to(device)
        else:
            image, target = batch
            lr = None
        image = even(image).to(device)
        target = even(target).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            prediction = model(image, lr) if lr is not None else model(image)
        prediction = prediction.float().clamp(0, 1)
        diff = prediction - target.float()
        mse_sum += diff.square().mean().item()
        mae_sum += diff.abs().mean().item()
        pdx = prediction[..., :, 1:] - prediction[..., :, :-1]
        pdy = prediction[..., 1:, :] - prediction[..., :-1, :]
        tdx = target.float()[..., :, 1:] - target.float()[..., :, :-1]
        tdy = target.float()[..., 1:, :] - target.float()[..., :-1, :]
        edge_sum += ((pdx - tdx).abs().mean() + (pdy - tdy).abs().mean()).item()
        count += 1
    model.train()
    mse, mae, edge = mse_sum / count, mae_sum / count, edge_sum / count
    return {"mse": mse, "mae": mae, "edge_mae": edge,
            "psnr_db": 10.0 * math.log10(1.0 / max(mse, 1e-12))}


def save(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def append_train_loss(path, iteration, loss, pixel, gradient, lr):
    row = {"iteration": iteration, "loss": loss, "pixel": pixel,
           "gradient": gradient, "lr": lr}
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--parallelism", choices=("ddp", "dp", "single"), default="ddp")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/d/ResearchCode/S2R-HDR/data/20260818"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, default=None,
                        help="Shared float32 NPY cache for decoded training TIFFs")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--batch-size-per-gpu", type=int, default=16)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--total-iter", type=int, default=60000)
    parser.add_argument("--stop-after", type=int, default=None,
                        help="Stop and checkpoint early while preserving the total-iter scheduler protocol")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--val-every", type=int, default=5000)
    parser.add_argument("--val-at-start", action="store_true",
                        help="Run full-image validation immediately after resume/init")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-lr-checkpoint", type=Path, default=None,
                        help="Exp E1 checkpoint used to initialize only the VIV-LR predictor")
    parser.add_argument("--init-model-checkpoint", type=Path, default=None,
                        help="Load a complete same-variant Exp E checkpoint into a new joint run")
    parser.add_argument("--freeze-lr", action="store_true",
                        help="Freeze the initialized VIV-LR predictor during affine ablations")
    parser.add_argument("--loss-mode", choices=LOSS_MODES, default="base")
    args = parser.parse_args()
    if args.grad_accumulation < 1:
        raise ValueError("grad-accumulation must be at least 1")
    if args.variant not in EXPE_VARIANTS and (args.init_lr_checkpoint or args.init_model_checkpoint or
                                              args.freeze_lr or args.loss_mode != "base"):
        raise ValueError("Exp E initialization and loss flags require an Exp E variant")
    if args.init_lr_checkpoint and args.init_model_checkpoint:
        raise ValueError("Choose only one initialization checkpoint")
    if args.resume and args.init_model_checkpoint and not args.init_model_checkpoint.is_file():
        raise FileNotFoundError(args.init_model_checkpoint)
    if args.init_model_checkpoint and args.freeze_lr:
        raise ValueError("Full-model joint initialization must leave LRNet trainable")
    if args.resume and args.init_lr_checkpoint and not args.init_lr_checkpoint.is_file():
        raise FileNotFoundError(args.init_lr_checkpoint)
    if args.freeze_lr and not (args.init_lr_checkpoint or args.resume):
        raise ValueError("--freeze-lr requires --init-lr-checkpoint or --resume")

    if args.parallelism == "ddp":
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError("DDP mode requires torchrun/torch.distributed.run with LOCAL_RANK")
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
            raise ValueError(f"Unsupported DDP backend: {backend}")
        dist.init_process_group(backend, init_method="env://")
        device = torch.device("cuda", local_rank)
        sampler_replicas, sampler_rank = world, rank
    elif args.parallelism == "dp":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
            raise RuntimeError(f"Exp C DataParallel requires four visible CUDA GPUs, found {torch.cuda.device_count()}")
        rank, world, local_rank = 0, 4, 0
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        sampler_replicas, sampler_rank = 1, 0
        backend = "data_parallel"
    else:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(f"Exp C single-GPU mode requires exactly one visible CUDA GPU, found {torch.cuda.device_count()}")
        rank, world, local_rank = 0, 1, 0
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        sampler_replicas, sampler_rank = 1, 0
        backend = "single_gpu"
    seed_all(args.seed, rank, args.parallelism)

    dataset = WLSPairs(args.data_root, "train")
    train_ids, val_ids = partition_ids(dataset.ids, args.val_count, args.seed)
    id_map = {key: index for index, key in enumerate(dataset.ids)}
    cache_path = args.cache_path or args.output_dir.parent / "train_0818_float32.npy"
    cache_path = ensure_float32_cache(dataset, cache_path, rank, args.parallelism)
    guided = args.variant in GUIDED_VARIANTS
    if guided:
        lr_cache = ensure_lr_cache(
            dataset, args.data_root, cache_path.with_name(cache_path.stem + "_lr.npy"),
            rank, args.parallelism,
        )
        train_set = RandomGuidedPatchPairs(
            cache_path, lr_cache, [id_map[key] for key in train_ids], args.crop_size
        )
        val_set = (GuidedTriplets(Subset(dataset, [id_map[key] for key in val_ids]),
                                  args.data_root, "train") if rank == 0 else None)
    else:
        train_set = RandomPatchPairs(cache_path, [id_map[key] for key in train_ids], args.crop_size)
        val_set = Subset(dataset, [id_map[key] for key in val_ids]) if rank == 0 else None
    loader_batch_size = args.batch_size_per_gpu * world if args.parallelism == "dp" else args.batch_size_per_gpu
    sampler = DistributedSampler(train_set, num_replicas=sampler_replicas, rank=sampler_rank,
                                 shuffle=True, seed=args.seed, drop_last=True)
    train_loader = DataLoader(
        train_set, batch_size=loader_batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_guided if guided else collate_pairs,
    ) if rank == 0 else None

    model = build_model(args.variant).to(device)
    if args.variant in EXPE_VARIANTS:
        if args.init_lr_checkpoint and not args.resume:
            initialize_lr(model, args.init_lr_checkpoint, freeze=args.freeze_lr)
        elif args.init_model_checkpoint and not args.resume:
            initialize_model(model, args.init_model_checkpoint, args.variant)
        elif args.freeze_lr:
            model.lr_frozen = True
            for parameter in model.lr.parameters():
                parameter.requires_grad_(False)
            model.lr.eval()
    if args.parallelism == "ddp":
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=(args.variant in EXPE_VARIANTS
                                            and args.variant != "exp7c_e0"))
    elif args.parallelism == "dp":
        model = torch.nn.DataParallel(model, device_ids=list(range(world)))
    model_core = model.module if args.parallelism in {"ddp", "dp"} else model
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_iter, eta_min=args.min_lr
    )
    run_protocol = protocol(args, train_ids, val_ids)
    start_iter, best_mse = 0, float("inf")
    latest_path = args.output_dir / "latest.pth"
    best_path = args.output_dir / "best.pth"
    history_path = args.output_dir / "history.csv"
    train_loss_path = args.output_dir / "train_loss.csv"
    if args.resume:
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        compatibility_fields = (set() if args.variant == "exp7c_e0" else
                                {"parallelism", "backend", "world_size", "batch_size_per_gpu",
                                 "gradient_accumulation", "effective_batch_size"})
        saved_protocol = {k: v for k, v in state["protocol"].items() if k not in compatibility_fields}
        current_protocol = {k: v for k, v in run_protocol.items() if k not in compatibility_fields}
        if saved_protocol != current_protocol:
            raise ValueError("Resume protocol does not match this run")
        model_core.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_iter, best_mse = state["iteration"], state["best_mse"]
    elif any(path.exists() for path in (latest_path, best_path, history_path, train_loss_path,
                                        args.output_dir / "protocol.json")):
        raise FileExistsError(f"Training artifacts already exist in {args.output_dir}; "
                              "use --resume or a new output directory")
    if args.parallelism == "ddp":
        ddp_barrier()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0 and not args.resume:
        (args.output_dir / "protocol.json").write_text(
            json.dumps(run_protocol, indent=2) + "\n", encoding="utf-8"
        )
    if rank == 0:
        print(f"Exp C {args.variant}: world={world}, batch={args.batch_size_per_gpu}x{world}, "
              f"backend={backend}, crop={args.crop_size}, iterations={args.total_iter}, device={device}, "
              f"workers={args.num_workers}, pin_memory={args.pin_memory}", flush=True)

    iteration = start_iter
    stop_iter = min(args.total_iter, args.stop_after) if args.stop_after else args.total_iter
    progress = tqdm(
        total=stop_iter - start_iter,
        desc=f"{args.variant} train",
        unit="it",
        dynamic_ncols=True,
        disable=rank != 0,
    )

    if args.val_at_start and rank == 0:
        metrics = validate(model_core, val_loader, device, args.precision)
        is_best = metrics["mse"] < best_mse
        if is_best:
            best_mse = metrics["mse"]
            save(best_path, {"iteration": start_iter, "metrics": metrics,
                             "model": model_core.state_dict(), "protocol": run_protocol})
        row = {"iteration": start_iter, **metrics, "best": int(is_best)}
        exists = history_path.exists()
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        progress.write(
            f"VALID {args.variant} iter={start_iter}: "
            f"{json.dumps(metrics)} best={is_best}"
        )

    iterator = iter(train_loader)
    epoch = 0
    model.train()
    while iteration < stop_iter:
        optimizer.zero_grad(set_to_none=True)
        loss_value = pixel_value = grad_value = 0.0
        for _ in range(args.grad_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(train_loader)
                batch = next(iterator)
            if len(batch) == 3:
                lr, image, target = batch
                lr = lr.to(device, non_blocking=args.pin_memory)
            else:
                image, target = batch
                lr = None
            image = image.to(device, non_blocking=args.pin_memory)
            target = target.to(device, non_blocking=args.pin_memory)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                if args.variant in EXPE_VARIANTS:
                    prediction, auxiliary = model(image, lr, return_aux=True)
                    loss, pixel_loss, grad_loss = experiment_loss(
                        prediction.float(), target.float(), auxiliary, args.loss_mode
                    )
                    if getattr(model_core, "route_supervision", False):
                        loss = loss + ROUTE_SUPERVISION_WEIGHT * route_supervision_loss(
                            auxiliary, image, target
                        )
                else:
                    prediction = model(image, lr) if lr is not None else model(image)
                    loss, pixel_loss, grad_loss = edge_loss(prediction.float(), target.float())
            (loss / args.grad_accumulation).backward()
            loss_value += loss.detach().item() / args.grad_accumulation
            pixel_value += pixel_loss.item() / args.grad_accumulation
            grad_value += grad_loss.item() / args.grad_accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        iteration += 1

        log_now = iteration == start_iter + 1 or iteration % 100 == 0 or iteration == stop_iter
        if rank == 0:
            progress.set_postfix(
                loss=f"{loss_value:.3g}",
                pixel=f"{pixel_value:.3g}",
                grad=f"{grad_value:.3g}",
                lr=f"{scheduler.get_last_lr()[0]:.2g}",
            )
            progress.update(1)
        if rank == 0 and log_now:
            append_train_loss(train_loss_path, iteration, loss_value, pixel_value,
                              grad_value, scheduler.get_last_lr()[0])
        validate_now = iteration % args.val_every == 0 or iteration == stop_iter
        save_now = iteration % args.save_every == 0 or iteration == stop_iter or validate_now
        if validate_now:
            if rank == 0:
                metrics = validate(model_core, val_loader, device, args.precision)
                is_best = metrics["mse"] < best_mse
                if is_best:
                    best_mse = metrics["mse"]
                    save(best_path, {"iteration": iteration, "metrics": metrics,
                                     "model": model_core.state_dict(), "protocol": run_protocol})
                row = {"iteration": iteration, **metrics, "best": int(is_best)}
                exists = history_path.exists()
                with history_path.open("a", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    if not exists:
                        writer.writeheader()
                    writer.writerow(row)
                progress.write(
                    f"VALID {args.variant} iter={iteration}: "
                    f"{json.dumps(metrics)} best={is_best}"
                )
            if args.parallelism == "ddp":
                ddp_barrier()
        if save_now:
            if rank == 0:
                save(latest_path, {"iteration": iteration, "best_mse": best_mse,
                     "model": model_core.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "protocol": run_protocol})
            if args.parallelism == "ddp":
                ddp_barrier()

    if rank == 0:
        progress.close()
        save(args.output_dir / "final.pth", {"iteration": iteration, "best_mse": best_mse,
             "model": model_core.state_dict(), "protocol": run_protocol})
    if args.parallelism == "ddp":
        ddp_barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
