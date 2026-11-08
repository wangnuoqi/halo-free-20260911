"""Verify the matched E runs and archive obsolete E checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "expe" / "runs"
OLD = RUNS / "legacy_matched_exposure3840000_v5"
E1 = RUNS / "scratch" / "viv_gf_e1" / "seed42_ddpfix"
E0_NEW = RUNS / "scratch" / "exp7c_e0" / "seed42_exposure3840000"
E0_NEW_STAGES = RUNS / "three_stage" / "exp7c_e0" / "seed42_exposure3840000"


def load_final(directory: Path, name: str, **expected):
    path = directory / name
    best = directory / name.replace("final.pth", "best.pth")
    if not path.is_file() or not best.is_file():
        raise FileNotFoundError(f"completed final/best checkpoint required in {directory}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    protocol = state.get("protocol", {})
    for key, value in expected.items():
        actual = state.get(key) if key in {"iteration", "epoch", "step", "phase"} else protocol.get(key)
        if actual != value:
            raise ValueError(f"{path}: {key}={actual!r}, expected {value!r}")
    print(f"[verified] {path}", flush=True)
    return state


def verify_baseline():
    state = load_final(E1, "final.pth", iteration=60000, world_size=4,
                       batch_size_per_gpu=16, gradient_accumulation=1,
                       effective_batch_size=64, parallelism="ddp", backend="nccl")
    return state["protocol"]


def verify_all():
    baseline = verify_baseline()
    checks = [
        (E0_NEW, "final.pth", dict(iteration=60000, world_size=2,
             batch_size_per_gpu=16, gradient_accumulation=2,
             effective_batch_size=64, parallelism="ddp", backend="nccl",
             train_crop_size=128, seed=42)),
        (E0_NEW_STAGES / "refine", "refine_final.pth",
         dict(phase="refine", epoch=200, step=5600, world_size=2,
              batch_size_per_rank=32, global_batch_size=64, num_workers=8,
              crop_size=128, seed=42, val_count=200, epochs=200)),
        (E0_NEW_STAGES / "joint", "joint_final.pth",
         dict(phase="joint", epoch=200, step=5600, world_size=2,
              batch_size_per_rank=32, global_batch_size=64, num_workers=8,
              crop_size=128, seed=42, val_count=200, epochs=200)),
    ]
    for slot, phases in (("e0", ("base", "refine", "joint")),
                         ("e1", ("base",))):
        for phase in phases:
            checks.append((OLD / slot / phase, "final.pth",
                           dict(phase=phase, iteration=60000, world_size=2,
                                batch_size_per_rank=16, gradient_accumulation=2,
                                global_batch_size=64, sample_exposures=3840000,
                                parallelism="ddp", backend="nccl", num_workers=8)))
    for directory, name, expected in checks:
        state = load_final(directory, name, **expected)
        protocol = state["protocol"]
        for digest_key in ("train_ids_sha256", "validation_ids_sha256"):
            if digest_key in protocol and protocol[digest_key] != baseline[digest_key]:
                raise ValueError(f"{directory}: {digest_key} differs from E1")
        if "train_ids" in protocol:
            for ids_key, digest_key in (("train_ids", "train_ids_sha256"),
                                        ("validation_ids", "validation_ids_sha256")):
                ids = [identifier.removesuffix("-log") for identifier in protocol[ids_key]]
                digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
                if digest != baseline[digest_key]:
                    raise ValueError(f"{directory}: {ids_key} differ from E1")
        elif not all(key in protocol for key in ("train_ids_sha256", "validation_ids_sha256")):
            raise ValueError(f"{directory}: missing data split hashes")
    return baseline


def archive_candidates():
    # Keep the completed E0 new-recipe run with unmatched exposure, and the
    # completed E1 new-recipe chain, as historical comparison checkpoints.
    candidates = [path for path in RUNS.glob("legacy_matched_patch128*") if path.is_dir()]
    return [path for path in candidates if path.is_dir() and any(path.rglob("*.pth"))]


def archive_old():
    candidates = archive_candidates()
    if not candidates:
        print("[archive] no obsolete E checkpoints remain", flush=True)
        return
    archive = ROOT / "expe" / "archive" / "legacy_e_checkpoints" / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
    archive.mkdir(parents=True, exist_ok=False)
    moved = []
    for source in candidates:
        relative = source.relative_to(RUNS)
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        moved.append({"from": str(source), "to": str(destination)})
        (archive / "manifest.json").write_text(json.dumps(moved, indent=2) + "\n", encoding="utf-8")
        print(f"[archived] {source} -> {destination}", flush=True)
    print(f"[archive] manifest: {archive / 'manifest.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--baseline-only", action="store_true")
    group.add_argument("--plan", action="store_true")
    group.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if args.plan:
        for path in archive_candidates():
            print(path)
    elif args.baseline_only:
        verify_baseline()
    else:
        verify_all()
        archive_old()


if __name__ == "__main__":
    main()
