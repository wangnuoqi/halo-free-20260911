"""Time clip-art exp3-b and VIVNet with matched 512px Q30 inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_all_models as evaluation


def cuda_times(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def wall_times(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - started))
    return times


def summarize(timings: dict[str, list[list[float]]]) -> dict:
    return {
        name: {
            "median_of_image_medians_ms": statistics.median(statistics.median(row) for row in values),
            "per_image_median_ms": [statistics.median(row) for row in values],
            "per_image_repeats_ms": values,
        }
        for name, values in timings.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--exp3b-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "expd/results/clipart_fp32.json")
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("samples and repeats must be positive; warmup must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    ids = [line.strip() for line in (ROOT / "openclipart_5000/test_list.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    paths = [ROOT / "openclipart_5000/jpeg_pairs_512/jpeg_q30" / f"{stem}.jpg" for stem in ids[:args.samples]]
    if len(paths) != args.samples or not all(path.is_file() for path in paths):
        raise RuntimeError("Requested test images are missing")

    ours = evaluation.load_ours("exp3-b", ROOT, device, checkpoint_override=args.exp3b_checkpoint)
    vivnet = evaluation.load_baselines(ROOT, device, ["VIVNet"])["VIVNet"]
    timings = {name: [] for name in ("exp3b_forward", "exp3b_base", "exp3b_refine", "vivnet_forward", "exp3b_pipeline", "vivnet_pipeline")}
    with torch.no_grad():
        for path in paths:
            rgb = evaluation.read_rgb(path)
            hr = evaluation.to_batch([rgb], device)
            lr = evaluation.to_batch([evaluation.resize_half_rgb(rgb)], device)

            def base():
                return ours["base"](lr - 0.5, hr - 0.5)

            coarse = base()

            def refine():
                return (ours["refine"](hr - 0.5, coarse - 0.5)["final"] + 0.5).clamp(0, 1)

            def full():
                prediction = ours["refine"](hr - 0.5, ours["base"](lr - 0.5, hr - 0.5) - 0.5)["final"]
                return (prediction + 0.5).clamp(0, 1)

            def viv():
                height, width = hr.shape[-2:]
                padded = F.pad(hr, (0, (-width) % 32, 0, (-height) % 32), mode="reflect")
                return vivnet(padded)[..., :height, :width].clamp(0, 1)

            functions = {
                "exp3b_forward": full,
                "exp3b_base": base,
                "exp3b_refine": refine,
                "vivnet_forward": viv,
            }
            for name, fn in functions.items():
                values = cuda_times(fn, args.warmup, args.repeats)
                timings[name].append(values)
                print(f"{path.stem} {name}: {statistics.median(values):.3f} ms", flush=True)
            for name, fn in {
                "exp3b_pipeline": lambda: evaluation.predict_ours(ours, [path], device),
                "vivnet_pipeline": lambda: evaluation.predict_baseline("VIVNet", vivnet, [path], device),
            }.items():
                values = wall_times(fn, args.warmup, args.repeats)
                timings[name].append(values)
                print(f"{path.stem} {name}: {statistics.median(values):.3f} ms", flush=True)

    report = {
        "protocol": {
            "dataset": "OpenClipart Q30 test", "ids": ids[:args.samples],
            "device": str(device), "gpu": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "precision": "fp32", "batch_size": 1,
            "warmup": args.warmup, "repeats": args.repeats,
            "forward_timing": "CUDA events, GPU tensors preloaded",
            "pipeline_timing": "synchronized wall time of evaluate_all_models.predict_* including JPEG decode, preprocessing, H2D and D2H",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "checkpoints": {
            "exp3b": str(ours["checkpoint"]),
            "vivnet": evaluation.BASELINE_CHECKPOINTS["VIVNet"],
        },
        "parameters": {
            "exp3b": sum(p.numel() for model in (ours["base"], ours["refine"]) for p in model.parameters()),
            "vivnet": sum(p.numel() for p in vivnet.parameters()),
        },
        "latency": summarize(timings),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
