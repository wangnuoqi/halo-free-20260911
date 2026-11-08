"""Measure each replica forward inside four-GPU DataParallel."""
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


class TimedForward(nn.Module):
    records = []

    def __init__(self, name, model):
        super().__init__()
        self.name = name
        self.model = model

    def forward(self, *inputs):
        device = inputs[0].device
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = self.model(*inputs)
        torch.cuda.synchronize(device)
        self.records.append({"name": self.name, "device": str(device),
                             "ms": 1000 * (time.perf_counter() - started)})
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "expd/results/dp_workers.json")
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
    base = nn.DataParallel(TimedForward("base", ours["base"]))
    refine = nn.DataParallel(TimedForward("refine", ours["refine"]))
    with torch.no_grad():
        for iteration in range(args.warmup + args.repeats):
            coarse = base(lr - 0.5, hr - 0.5)
            refine(hr - 0.5, coarse - 0.5)
            if iteration == args.warmup - 1:
                TimedForward.records.clear()
    grouped = {}
    for record in TimedForward.records:
        grouped.setdefault(record["name"], {}).setdefault(record["device"], []).append(record["ms"])
    report = {
        "protocol": {"checkpoint": str(ours["checkpoint"]), "batch": 4,
                     "gpus": 4, "warmup": args.warmup, "repeats": args.repeats,
                     "timing": "synchronized wall time inside each replica forward"},
        "median_ms": {name: {gpu: statistics.median(values) for gpu, values in devices.items()}
                      for name, devices in grouped.items()},
        "records": grouped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["median_ms"], indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
