"""Time the same clip-art model sequentially on each GPU without DataParallel."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_all_models as evaluation


def measure(fn, device, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        values.append(1000 * (time.perf_counter() - started))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "expd/results/direct_gpus.json")
    args = parser.parse_args()
    if torch.cuda.device_count() != 4:
        raise RuntimeError("Expose exactly four GPUs")
    ours = evaluation.load_ours("exp3-b", ROOT, torch.device("cuda:0"))
    ids = (ROOT / "openclipart_5000/test_list.txt").read_text(encoding="utf-8").splitlines()[:4]
    paths = [ROOT / "openclipart_5000/jpeg_pairs_512/jpeg_q30" / f"{stem}.jpg" for stem in ids]
    rgb = [evaluation.read_rgb(path) for path in paths]
    result = {}
    with torch.no_grad():
        for index, image in enumerate(rgb):
            device = torch.device(f"cuda:{index}")
            base = copy.deepcopy(ours["base"]).to(device).eval()
            refine = copy.deepcopy(ours["refine"]).to(device).eval()
            hr = evaluation.to_batch([image], device)
            lr = evaluation.to_batch([evaluation.resize_half_rgb(image)], device)
            x = hr - 0.5
            x_lr = lr - 0.5
            coarse = base(x_lr, x)
            functions = {
                "base": lambda: base(x_lr, x),
                "refine": lambda: refine(x, coarse - 0.5)["final"],
                "full": lambda: refine(x, base(x_lr, x) - 0.5)["final"],
            }
            result[str(device)] = {}
            for name, fn in functions.items():
                values = measure(fn, device, args.warmup, args.repeats)
                result[str(device)][name] = {"median_ms": statistics.median(values), "all_ms": values}
                print(f"{device} {name}: {statistics.median(values):.3f} ms", flush=True)
            del base, refine
    report = {"protocol": {"checkpoint": str(ours["checkpoint"]), "precision": "fp32",
                           "batch": 1, "warmup": args.warmup, "repeats": args.repeats,
                           "timing": "synchronized wall time; four GPUs executed sequentially"},
              "latency": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
