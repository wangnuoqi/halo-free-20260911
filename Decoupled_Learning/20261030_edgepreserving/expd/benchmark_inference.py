"""Compare exp7C, NAFNet, and VIVNet forward latency on identical WLS inputs."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import cv2
import torch


HERE = Path(__file__).resolve().parent
BASELINE_ROOT = HERE.parents[1] / "20260815_edgepreserving"
BASELINE = BASELINE_ROOT / "baseline"
DATA_ROOT = Path(r"D:\ResearchCode\S2R-HDR\data\20260818")
EXP7C_ROOT = (BASELINE_ROOT / "experiments/exp7/exp7C_20260818_joint_fidelity"
              / "models/exp7C_20260818_joint_fidelity")
sys.path.insert(0, str(BASELINE_ROOT))

from baseline.wls_data import WLSPairs, _files_by_id
from baseline.evaluate_nafnet_wls import build_model as build_nafnet
from baseline.vivnet_wls.model import build_model as build_vivnet, run_padded
from evaluate_exp7c_metrics import load_model_definitions, load_state


def even(image: torch.Tensor) -> torch.Tensor:
    return image[..., :image.shape[-2] - image.shape[-2] % 2,
                 :image.shape[-1] - image.shape[-1] % 2]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def measure(fn, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        if not torch.isfinite(output).all().item():
            raise RuntimeError("Non-finite model output during latency benchmark")
        timings.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(timings),
        "p10_ms": percentile(timings, 0.1),
        "p90_ms": percentile(timings, 0.9),
        "all_ms": timings,
    }


def model_parameters(*models: torch.nn.Module) -> int:
    return sum(parameter.numel() for model in models for parameter in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "inference_fp32.json")
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("samples and repeats must be positive; warmup must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for comparable GPU latency")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dataset = WLSPairs(args.data_root, "test")
    lowres_by_id = _files_by_id(args.data_root / "testinput-l0p500-a1p400-200_downsample", "-log")
    if set(dataset.ids) != set(lowres_by_id):
        raise ValueError("exp7C low-resolution inputs do not match baseline test IDs")
    indices = sorted({round(i * (len(dataset) - 1) / max(args.samples - 1, 1))
                      for i in range(args.samples)})

    exp7_source = load_model_definitions(BASELINE_ROOT / "experiments/exp7C_joint_fidelity_20260818.py")
    conv = exp7_source["DeepGuidedFilter"](radius=1, eps=1e-8).to(device).eval()
    refine = exp7_source["CascadeRefine3Stage"](
        base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True
    ).to(device).eval()
    load_state(conv, EXP7C_ROOT / "convguided_joint_best_20260818_l0p500_a1p400.pth", device)
    load_state(refine, EXP7C_ROOT / "rf_3stage_best_20260818_l0p500_a1p400.pth", device)

    nafnet = build_nafnet(device).eval()
    naf_path = BASELINE / "runs/nafnet_wls_patch256_nccl_3gpu_bs32/nafnet_wls_best.pth"
    naf_state = torch.load(naf_path, map_location="cpu", weights_only=False)
    nafnet.load_state_dict(naf_state["model_state"], strict=True)
    del naf_state

    vivnet = build_vivnet(device).eval()
    viv_path = BASELINE / "runs/vivnet_wls_patch128_nccl_4gpu_bs48/vivnet_wls_best.pth"
    viv_state = torch.load(viv_path, map_location="cpu", weights_only=False)
    vivnet.load_state_dict(viv_state["model_state"], strict=True)
    del viv_state

    measurements = {name: [] for name in ("exp7c_total", "exp7c_coarse", "exp7c_refine", "nafnet", "vivnet")}
    image_shapes = {}
    with torch.inference_mode():
        for index in indices:
            identifier = dataset.ids[index]
            image, _ = dataset[index]
            lr = cv2.imread(str(lowres_by_id[identifier]), cv2.IMREAD_UNCHANGED)
            if lr is None or lr.ndim != 2:
                raise ValueError(f"Invalid exp7C low-resolution input: {identifier}")
            image = even(image[None]).to(device)
            image_shapes[identifier] = list(image.shape[-2:])
            lowres = torch.from_numpy(lr.astype("float32"))[None, None].to(device)
            x = image - 0.5
            x_lr = lowres - 0.5

            def coarse():
                return conv(x_lr, x)

            coarse_input = coarse()

            def refine_only():
                return refine(x, coarse_input - 0.5)["final"]

            def exp7c_total():
                return (refine(x, conv(x_lr, x) - 0.5)["final"] + 0.5).clamp(0.0, 1.0)

            functions = {
                "exp7c_total": exp7c_total,
                "exp7c_coarse": coarse,
                "exp7c_refine": refine_only,
                "nafnet": lambda: nafnet(image).clamp(0.0, 1.0),
                "vivnet": lambda: run_padded(vivnet, image).clamp(0.0, 1.0),
            }
            for name, fn in functions.items():
                result = measure(fn, args.warmup, args.repeats)
                measurements[name].append({"image_id": identifier, **result})
                print(f"{identifier} {name}: median={result['median_ms']:.3f} ms", flush=True)

    summary = {
        name: {
            "median_ms": statistics.median(item["median_ms"] for item in values),
            "per_image": values,
        }
        for name, values in measurements.items()
    }
    report = {
        "protocol": {
            "device": str(device), "gpu": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "precision": "fp32", "batch_size": 1, "test_ids": [dataset.ids[i] for i in indices],
            "image_shapes_hw": image_shapes,
            "image_geometry": "full_image_even_crop", "warmup": args.warmup,
            "repeats_per_image": args.repeats,
            "timing": "CUDA events around GPU forward only; inputs preloaded on device",
            "concurrent_gpu_jobs_possible": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "parameters": {
            "exp7c": model_parameters(conv, refine),
            "nafnet": model_parameters(nafnet),
            "vivnet": model_parameters(vivnet),
        },
        "latency": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
