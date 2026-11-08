"""Run four Exp C four-GPU DataParallel experiments sequentially in WinTrain."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA_ROOT = Path(r"D:\ResearchCode\S2R-HDR\data\20260818")
VARIANTS = (
    "naf_small",
    "similarity_pyramid",
    "lowres_edge",
    "unrolled_diffusion",
    "wls_naf",
    "deep_guided_lr4",
    "deep_guided_lr8",
    "guided_pyramid_lr4",
)
SCREENING_VARIANTS = VARIANTS[:4]


def run(command, env, cwd):
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] Running: {' '.join(map(str, command))}", flush=True)
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def train(script, variant, output, data_root, env, *, batch, iterations,
          val_count, val_every, save_every, workers, cache_path, resume=False,
          pin_memory=False, validate_at_start=False, stop_after=None):
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", str(script), "--parallelism", "dp",
        "--variant", variant, "--data-root", str(data_root),
        "--output-dir", str(output), "--cache-path", str(cache_path), "--crop-size", "128",
        "--batch-size-per-gpu", str(batch), "--total-iter", str(iterations),
        "--val-count", str(val_count), "--val-every", str(val_every),
        "--save-every", str(save_every), "--num-workers", str(workers),
        "--precision", "bf16", "--lr", "3e-4", "--min-lr", "1e-6",
    ]
    if resume and (output / "latest.pth").exists():
        command.append("--resume")
    if pin_memory:
        command.append("--pin-memory")
    if validate_at_start:
        command.append("--val-at-start")
    if stop_after is not None:
        command.extend(("--stop-after", str(stop_after)))
    run(command, env, PROJECT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=HERE / "runs" / "0818_four_hypotheses_4gpu")
    parser.add_argument("--evaluation-root", type=Path, default=HERE / "evaluation" / "test_20260818")
    parser.add_argument("--formal-iterations", type=int, default=60000)
    parser.add_argument("--screening-iterations", type=int, default=5000,
                        help="Common first-pass budget for all four hypotheses")
    parser.add_argument("--screening-only", action="store_true",
                        help="Train/evaluate each candidate to the screening budget, then exit")
    parser.add_argument("--parallel-candidates", action="store_true",
                        help="Run the four candidates concurrently, one single-GPU process per GPU")
    parser.add_argument("--variant", choices=VARIANTS, default=None,
                        help="Run one candidate in its own WinTrain process")
    parser.add_argument("--batch-per-gpu", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=100,
                        help="Checkpoint interval for formal runs")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Run the four short four-GPU smoke tests and exit")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip smoke tests that have already passed")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Windows DataLoader workers reading crops from the shared map")
    parser.add_argument("--pin-memory", action="store_true",
                        help="Pin training batches before host-to-device transfer")
    args = parser.parse_args()
    args.data_root = args.data_root if args.data_root.is_absolute() else (HERE / args.data_root).resolve()
    args.run_root = args.run_root if args.run_root.is_absolute() else (HERE / args.run_root).resolve()
    args.evaluation_root = (
        args.evaluation_root if args.evaluation_root.is_absolute()
        else (HERE / args.evaluation_root).resolve()
    )
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.evaluation_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
    })
    print("Exp C WinTrain sequence: four GPUs, DataParallel, 0818 WLS data", flush=True)
    print(f"Data: {args.data_root}; outputs: {args.run_root}", flush=True)

    train_script = HERE / "src" / "train_expc.py"
    evaluate_script = HERE / "src" / "evaluate_expc.py"
    shared_cache = HERE / "runs" / "0818_memmap_4gpu" / "train_0818_float32.npy"
    cache_path = shared_cache if shared_cache.exists() else args.run_root / "train_0818_float32.npy"
    if args.parallel_candidates:
        processes = []
        for gpu, variant in enumerate(SCREENING_VARIANTS):
            output = args.run_root / variant
            output.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, "-u", str(train_script), "--parallelism", "single",
                "--variant", variant, "--data-root", str(args.data_root),
                "--output-dir", str(output), "--cache-path", str(cache_path),
                "--crop-size", "128", "--batch-size-per-gpu", str(args.batch_per_gpu),
                "--grad-accumulation", "4",
                "--total-iter", str(args.formal_iterations), "--val-count", "200",
                "--val-every", str(args.screening_iterations), "--save-every", str(args.save_every),
                "--num-workers", "0", "--precision", "bf16", "--lr", "3e-4", "--min-lr", "1e-6",
                "--stop-after", str(args.screening_iterations),
            ]
            if args.pin_memory:
                command.append("--pin-memory")
            if (output / "latest.pth").exists():
                command.append("--resume")
            child_env = env.copy()
            child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = output / "screening.log"
            log_stream = log_path.open("a", encoding="utf-8", buffering=1)
            process = subprocess.Popen(command, cwd=str(PROJECT), env=child_env,
                                       stdout=log_stream, stderr=subprocess.STDOUT)
            processes.append((variant, process, log_stream))
            print(f"Started {variant} on GPU {gpu}, PID={process.pid}", flush=True)
        failures = []
        for variant, process, log_stream in processes:
            code = process.wait()
            log_stream.close()
            print(f"Finished screening {variant}: exit={code}", flush=True)
            if code:
                failures.append((variant, code))
        if failures:
            raise RuntimeError(f"Screening failures: {failures}")
        print("All four single-GPU screening runs completed; test set remains untouched.", flush=True)
        return

    if args.variant:
        output = args.run_root / args.variant
        train(train_script, args.variant, output, args.data_root, env,
              batch=args.batch_per_gpu, iterations=args.formal_iterations,
              val_count=200, val_every=args.screening_iterations,
              save_every=args.save_every, workers=args.num_workers,
              cache_path=cache_path, resume=True, pin_memory=args.pin_memory,
              stop_after=args.screening_iterations if args.screening_only else None)
        stage = "Screening completed" if args.screening_only else "Training completed"
        print(f"{stage} for {args.variant}; test set remains untouched.", flush=True)
        return

    if not args.skip_smoke:
        for variant in VARIANTS:
            smoke_output = args.run_root / "smoke" / variant
            train(train_script, variant, smoke_output, args.data_root, env,
                     batch=2, iterations=12, val_count=8, val_every=12,
                     save_every=12, workers=args.num_workers, cache_path=cache_path,
                     pin_memory=args.pin_memory)
        if args.smoke_only:
            print("All four Exp C smoke tests completed.", flush=True)
            return

    stop_after = args.screening_iterations if args.screening_only else None
    for variant in VARIANTS:
        output = args.run_root / variant
        train(train_script, variant, output, args.data_root, env,
                 batch=args.batch_per_gpu, iterations=args.formal_iterations,
                 val_count=200, val_every=args.screening_iterations, save_every=args.save_every,
                 workers=args.num_workers, cache_path=cache_path, resume=True,
                 pin_memory=args.pin_memory,
                 validate_at_start=False, stop_after=stop_after)
        if args.screening_only:
            print(f"Screening complete for {variant} at {args.screening_iterations} updates; "
                  "test set remains untouched.", flush=True)
            continue
        evaluate_command = [
            sys.executable, "-u", str(evaluate_script), "--checkpoint",
            str(output / "best.pth"), "--data-root", str(args.data_root),
            "--output-dir", str(args.evaluation_root / variant),
            "--device", "cuda:0", "--precision", "fp32",
        ]
        run(evaluate_command, env, PROJECT)
    if args.screening_only:
        print("All four screening runs and test evaluations completed.", flush=True)
        return

    benchmark_script = HERE / "src" / "benchmark_references.py"
    run([sys.executable, "-u", str(benchmark_script)], env, PROJECT)
    print("All four experiments and reference benchmarks completed.", flush=True)


if __name__ == "__main__":
    main()
