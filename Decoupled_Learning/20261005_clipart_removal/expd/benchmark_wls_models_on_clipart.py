"""Compare WLS-trained exp7C, NAFNet and VIVNet on clip-art JPEG luminance."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from PIL import Image
import torch


HERE = Path(__file__).resolve().parent
CLIPART_ROOT = HERE.parent
WLS_ROOT = HERE.parents[1] / "20260815_edgepreserving"
sys.path.insert(0, str(WLS_ROOT))

from baseline.evaluate_nafnet_wls import build_model as build_nafnet
from baseline.vivnet_wls.model import build_model as build_vivnet, run_padded
from evaluate_exp7c_metrics import load_model_definitions, load_state


def measure(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=HERE / "results/wls_models_on_clipart_fp32.json")
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("samples and repeats must be positive; warmup must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    definitions = load_model_definitions(WLS_ROOT / "experiments/exp7C_joint_fidelity_20260818.py")
    coarse_model = definitions["DeepGuidedFilter"](radius=1, eps=1e-8).to(device).eval()
    refine_model = definitions["CascadeRefine3Stage"](
        base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True
    ).to(device).eval()
    exp7c_dir = WLS_ROOT / "experiments/exp7/exp7C_20260818_joint_fidelity/models/exp7C_20260818_joint_fidelity"
    coarse_path = exp7c_dir / "convguided_joint_best_20260818_l0p500_a1p400.pth"
    refine_path = exp7c_dir / "rf_3stage_best_20260818_l0p500_a1p400.pth"
    load_state(coarse_model, coarse_path, device)
    load_state(refine_model, refine_path, device)

    nafnet = build_nafnet(device).eval()
    naf_path = WLS_ROOT / "baseline/runs/nafnet_wls_patch256_nccl_3gpu_bs32/nafnet_wls_best.pth"
    nafnet.load_state_dict(torch.load(naf_path, map_location="cpu", weights_only=False)["model_state"], strict=True)
    vivnet = build_vivnet(device).eval()
    viv_path = WLS_ROOT / "baseline/runs/vivnet_wls_patch128_nccl_4gpu_bs48/vivnet_wls_best.pth"
    vivnet.load_state_dict(torch.load(viv_path, map_location="cpu", weights_only=False)["model_state"], strict=True)

    ids = [line.strip() for line in (CLIPART_ROOT / "openclipart_5000/test_list.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = ids[:args.samples]
    if len(selected) != args.samples:
        raise ValueError("Not enough clip-art test images")
    values = {name: [] for name in ("exp7c_total", "exp7c_coarse", "exp7c_refine", "nafnet", "vivnet")}
    with torch.inference_mode():
        for stem in selected:
            path = CLIPART_ROOT / "openclipart_5000/jpeg_pairs_512/jpeg_q30" / f"{stem}.jpg"
            with Image.open(path) as image:
                y = np.asarray(image.convert("YCbCr"), dtype=np.float32)[..., 0] / 255.0
            if y.shape != (512, 512):
                raise ValueError(f"Expected 512x512 image: {path}")
            lr_y = np.asarray(
                Image.fromarray(np.clip(y * 255, 0, 255).astype(np.uint8), mode="L").resize(
                    (256, 256), Image.Resampling.LANCZOS
                ), dtype=np.float32
            ) / 255.0
            hr = torch.from_numpy(y.copy())[None, None].to(device)
            lr = torch.from_numpy(lr_y.copy())[None, None].to(device)
            x = hr - 0.5
            x_lr = lr - 0.5

            def coarse():
                return coarse_model(x_lr, x)

            coarse_input = coarse()

            def refine():
                return (refine_model(x, coarse_input - 0.5)["final"] + 0.5).clamp(0, 1)

            functions = {
                "exp7c_total": lambda: (refine_model(x, coarse_model(x_lr, x) - 0.5)["final"] + 0.5).clamp(0, 1),
                "exp7c_coarse": coarse,
                "exp7c_refine": refine,
                "nafnet": lambda: nafnet(hr).clamp(0, 1),
                "vivnet": lambda: run_padded(vivnet, hr).clamp(0, 1),
            }
            for name, fn in functions.items():
                timings = measure(fn, args.warmup, args.repeats)
                values[name].append(timings)
                print(f"{stem} {name}: {statistics.median(timings):.3f} ms", flush=True)

    report = {
        "protocol": {
            "input": "OpenClipart Q30 test JPEG Y channel, 512x512",
            "model_training": "All three checkpoints were trained on WLS, not clip-art",
            "ids": selected, "precision": "fp32", "batch_size": 1,
            "device": str(device), "gpu": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "warmup": args.warmup, "repeats": args.repeats,
            "timing": "CUDA events around GPU forward; preprocessed tensors preloaded on GPU",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "checkpoints": {"exp7c_coarse": str(coarse_path), "exp7c_refine": str(refine_path),
                        "nafnet": str(naf_path), "vivnet": str(viv_path)},
        "parameters": {
            "exp7c": sum(p.numel() for model in (coarse_model, refine_model) for p in model.parameters()),
            "nafnet": sum(p.numel() for p in nafnet.parameters()),
            "vivnet": sum(p.numel() for p in vivnet.parameters()),
        },
        "latency": {
            name: {"median_of_image_medians_ms": statistics.median(statistics.median(row) for row in rows),
                   "per_image_median_ms": [statistics.median(row) for row in rows],
                   "per_image_repeats_ms": rows}
            for name, rows in values.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
