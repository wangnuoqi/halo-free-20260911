"""Measure where Windows DataParallel spends time for clip-art exp3-b."""
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


def sync_all():
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


class TimedDataParallel(nn.DataParallel):
    def __init__(self, module):
        super().__init__(module)
        self.phases = {name: [] for name in ("scatter", "replicate", "parallel_apply", "gather")}

    def _record(self, name, fn, *args, **kwargs):
        sync_all()
        started = time.perf_counter()
        result = fn(*args, **kwargs)
        sync_all()
        self.phases[name].append(1000 * (time.perf_counter() - started))
        return result

    def scatter(self, inputs, kwargs, device_ids):
        return self._record("scatter", super().scatter, inputs, kwargs, device_ids)

    def replicate(self, module, device_ids):
        return self._record("replicate", super().replicate, module, device_ids)

    def parallel_apply(self, replicas, inputs, kwargs):
        return self._record("parallel_apply", super().parallel_apply, replicas, inputs, kwargs)

    def gather(self, outputs, output_device):
        return self._record("gather", super().gather, outputs, output_device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "expd/results/dp_phases.json")
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
    base = TimedDataParallel(ours["base"])
    refine = TimedDataParallel(ours["refine"])
    with torch.no_grad():
        for _ in range(args.warmup + args.repeats):
            coarse = base(lr - 0.5, hr - 0.5)
            refine(hr - 0.5, coarse - 0.5)["final"]
    report = {
        "protocol": {"checkpoint": str(ours["checkpoint"]), "batch": 4,
                     "gpus": 4, "warmup": args.warmup, "repeats": args.repeats,
                     "phase_timing": "synchronized wall time inside DataParallel methods"},
        "phases": {
            name: {
                phase: {"median_ms": statistics.median(values[args.warmup:]),
                        "all_ms": values[args.warmup:]}
                for phase, values in model.phases.items()
            }
            for name, model in (("base", base), ("refine", refine))
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, phases in report["phases"].items():
        print(name, {key: round(value["median_ms"], 3) for key, value in phases.items()}, flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
