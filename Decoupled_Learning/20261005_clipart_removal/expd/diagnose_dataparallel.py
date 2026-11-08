"""Isolate DataParallel scatter/gather costs in the clip-art cascade."""
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


class FinalOnly(nn.Module):
    def __init__(self, refine: nn.Module):
        super().__init__()
        self.refine = refine

    def forward(self, x, coarse):
        return self.refine(x, coarse)["final"]


class FusedFinal(nn.Module):
    def __init__(self, base: nn.Module, refine: nn.Module):
        super().__init__()
        self.base = base
        self.refine = refine

    def forward(self, lr, hr):
        x = hr - 0.5
        coarse = self.base(lr - 0.5, x)
        return self.refine(x, coarse - 0.5)["final"]


def synchronize_all():
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def measure(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    synchronize_all()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        synchronize_all()
        samples.append(1000 * (time.perf_counter() - started))
    return {"median_ms_per_batch4": statistics.median(samples), "all_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "expd/results/dataparallel_diagnosis.json")
    args = parser.parse_args()
    if torch.cuda.device_count() != 4:
        raise RuntimeError("Expose exactly four GPUs to reproduce the original evaluation")
    device = torch.device("cuda:0")
    ours = evaluation.load_ours("exp3-b", ROOT, device)
    base, refine = ours["base"], ours["refine"]
    base_dp = nn.DataParallel(base)
    refine_dp = nn.DataParallel(refine)
    final_dp = nn.DataParallel(FinalOnly(refine))
    fused_dp = nn.DataParallel(FusedFinal(base, refine))

    ids = [line.strip() for line in (ROOT / "openclipart_5000/test_list.txt").read_text(encoding="utf-8").splitlines() if line.strip()][:4]
    paths = [ROOT / "openclipart_5000/jpeg_pairs_512/jpeg_q30" / f"{stem}.jpg" for stem in ids]
    rgb = [evaluation.read_rgb(path) for path in paths]
    hr = evaluation.to_batch(rgb, device)
    lr = evaluation.to_batch([evaluation.resize_half_rgb(image) for image in rgb], device)
    x = hr - 0.5
    x_lr = lr - 0.5

    with torch.no_grad():
        coarse_single = base(x_lr, x)

        def single():
            return refine(x, base(x_lr, x) - 0.5)["final"]

        def two_dp():
            return refine_dp(x, base_dp(x_lr, x) - 0.5)["final"]

        def final_only_dp():
            return final_dp(x, base_dp(x_lr, x) - 0.5)

        functions = {
            "base_single": lambda: base(x_lr, x),
            "base_dp": lambda: base_dp(x_lr, x),
            "refine_single": lambda: refine(x, coarse_single - 0.5)["final"],
            "refine_dp_all_outputs": lambda: refine_dp(x, coarse_single - 0.5)["final"],
            "refine_dp_final_only": lambda: final_dp(x, coarse_single - 0.5),
            "full_single": single,
            "full_two_dp_all_outputs": two_dp,
            "full_two_dp_final_only": final_only_dp,
            "full_fused_dp_final_only": lambda: fused_dp(lr, hr),
        }
        reference = single()
        checks = {
            name: float((fn() - reference).abs().max().item())
            for name, fn in functions.items() if name.startswith("full_")
        }
        if max(checks.values()) > 1e-5:
            raise RuntimeError(f"Different model outputs: {checks}")
        timing = {}
        for name, fn in functions.items():
            timing[name] = measure(fn, args.warmup, args.repeats)
            print(f"{name}: {timing[name]['median_ms_per_batch4']:.3f} ms/batch", flush=True)

    report = {
        "protocol": {"checkpoint": str(ours["checkpoint"]), "test_ids": ids,
                     "shape": list(hr.shape), "precision": "fp32", "warmup": args.warmup,
                     "repeats": args.repeats, "visible_gpus": torch.cuda.device_count(),
                     "timing": "synchronized wall time, batch of four preloaded on cuda:0"},
        "max_output_difference": checks,
        "latency": timing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
