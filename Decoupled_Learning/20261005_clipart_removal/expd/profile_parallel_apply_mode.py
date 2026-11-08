"""Compare threaded and sequential DataParallel replica execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_all_models as evaluation


class SequentialApplyDataParallel(nn.DataParallel):
    def parallel_apply(self, replicas, inputs, kwargs):
        outputs = []
        for replica, arguments, keyword_arguments in zip(replicas, inputs, kwargs):
            with torch.cuda.device(arguments[0].device):
                outputs.append(replica(*arguments, **keyword_arguments))
        return outputs


def synchronize_all():
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def measure(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    synchronize_all()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        synchronize_all()
        times.append(1000 * (time.perf_counter() - started))
    return {"median_ms_per_batch4": statistics.median(times), "all_ms": times}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "expd/results/parallel_apply_modes.json")
    args = parser.parse_args()
    if torch.cuda.device_count() != 4:
        raise RuntimeError("Expose exactly four GPUs")
    device = torch.device("cuda:0")
    ours = evaluation.load_ours("exp3-b", ROOT, device)
    ids = (ROOT / "openclipart_5000/test_list.txt").read_text(encoding="utf-8").splitlines()[:4]
    paths = [ROOT / "openclipart_5000/jpeg_pairs_512/jpeg_q30" / f"{stem}.jpg" for stem in ids]
    rgb = [evaluation.read_rgb(path) for path in paths]
    hr = evaluation.to_batch(rgb, device)
    lr = evaluation.to_batch([evaluation.resize_half_rgb(image) for image in rgb], device)
    base_threaded = nn.DataParallel(ours["base"])
    refine_threaded = nn.DataParallel(ours["refine"])
    base_sequential = SequentialApplyDataParallel(ours["base"])
    refine_sequential = SequentialApplyDataParallel(ours["refine"])

    with torch.no_grad():
        def threaded():
            coarse = base_threaded(lr - 0.5, hr - 0.5)
            return refine_threaded(hr - 0.5, coarse - 0.5)["final"]

        def sequential():
            coarse = base_sequential(lr - 0.5, hr - 0.5)
            return refine_sequential(hr - 0.5, coarse - 0.5)["final"]

        difference = float((threaded() - sequential()).abs().max().item())
        if difference > 1e-5:
            raise RuntimeError(f"Outputs differ: {difference}")
        timing = {"threaded": measure(threaded, args.warmup, args.repeats),
                  "sequential": measure(sequential, args.warmup, args.repeats)}
    report = {"protocol": {"checkpoint": str(ours["checkpoint"]), "batch": 4,
                           "gpus": 4, "warmup": args.warmup, "repeats": args.repeats,
                           "timing": "synchronized wall time with same scatter, replicate and gather"},
              "max_output_difference": difference, "latency": timing}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print({name: round(item["median_ms_per_batch4"], 3) for name, item in timing.items()}, flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
