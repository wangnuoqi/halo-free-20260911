"""Run Experiment A variants sequentially with Windows torchrun + DDP/Gloo."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "exp_a" / "src" / "train_exp_a.py"
DEFAULT_DATA_ROOT = Path(r"D:\ResearchCode\S2R-HDR\data\20260818")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["a1", "c1", "a0"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size-per-gpu", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="short smoke test; 0 means full training")
    parser.add_argument("--master-port", type=int, default=29561)
    return parser.parse_args()


def main():
    args = parse_args()
    allowed = {"a0", "a1", "c0", "c1"}
    invalid = [variant for variant in args.variants if variant not in allowed]
    if invalid:
        raise SystemExit(f"Unsupported variants: {invalid}; choose from {sorted(allowed)}")
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise SystemExit(f"Expected four distinct GPU IDs, got: {args.gpus}")
    if args.batch_size_per_gpu < 1 or args.crop_size < 1 or args.epochs < 1:
        raise SystemExit("batch size, crop size, and epochs must be positive")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.run_root or (PROJECT_ROOT / "exp_a" / "runs" / f"windows_gloo_sequence_{stamp}")
    run_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "parallelism": "DistributedDataParallel",
        "backend": "gloo",
        "world_size": 4,
        "gpus": gpu_ids,
        "variants": args.variants,
        "data_root": str(args.data_root),
        "batch_size_per_gpu": args.batch_size_per_gpu,
        "global_batch_size": 4 * args.batch_size_per_gpu,
        "crop_size_hr": args.crop_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "precision": args.precision,
        "num_workers": args.num_workers,
        "max_steps": args.max_steps,
        "from_scratch_each_variant": True,
    }
    (run_root / "sequence_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )

    print(
        "Windows DDP/Gloo sequence: " + " -> ".join(args.variants),
        flush=True,
    )
    print(
        f"GPUs={','.join(gpu_ids)}; batch/GPU={args.batch_size_per_gpu}; "
        f"global batch={4 * args.batch_size_per_gpu}; HR crop={args.crop_size}; "
        f"epochs={args.epochs}; lr={args.lr}; precision={args.precision}; "
        f"workers={args.num_workers}; max_steps={args.max_steps}",
        flush=True,
    )
    print(f"Run root: {run_root}", flush=True)

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids),
        "TORCH_DDP_BACKEND": "gloo",
        "MASTER_ADDR": "127.0.0.1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
    })

    for index, variant in enumerate(args.variants):
        output_dir = run_root / variant
        port = args.master_port + index
        command = [
            sys.executable, "-u", "-m", "torch.distributed.run",
            "--standalone", "--nproc_per_node=4", f"--master_port={port}",
            str(TRAIN_SCRIPT),
            "--variant", variant,
            "--data-root", str(args.data_root),
            "--output-dir", str(output_dir),
            "--batch-size-per-gpu", str(args.batch_size_per_gpu),
            "--crop-size", str(args.crop_size),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--precision", args.precision,
            "--num-workers", str(args.num_workers),
            "--max-steps", str(args.max_steps),
        ]
        print(f"\n[{datetime.now().isoformat(timespec='seconds')}] Starting {variant}", flush=True)
        print("Command: " + subprocess.list2cmdline(command), flush=True)
        started = time.perf_counter()
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
        elapsed = time.perf_counter() - started
        if result.returncode != 0:
            raise SystemExit(f"{variant} failed with exit code {result.returncode}; stopping sequence")
        print(f"[{datetime.now().isoformat(timespec='seconds')}] Finished {variant}; elapsed={elapsed:.1f}s", flush=True)

    print(f"All variants completed. Checkpoints and configs are under {run_root}", flush=True)


if __name__ == "__main__":
    main()
