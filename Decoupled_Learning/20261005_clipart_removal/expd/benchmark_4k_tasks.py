"""Single-GPU 4K latency and parameter comparison across three tasks.

Inputs are representative held-out task images resized to 3840x2160. Timings
measure synchronized FP32 model inference on a preloaded GPU tensor; image I/O,
resize, host/device transfer, and checkpoint loading are excluded.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CLIPART = HERE.parent
DL_ROOT = HERE.parents[1]
EDGE = DL_ROOT / "20260815_edgepreserving"
TEXTURE = DL_ROOT / "20261008_textureremove"
sys.path.insert(0, str(EDGE))
sys.path.insert(0, str(TEXTURE))


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_state(model, path: Path, key=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if key:
        state = state[key]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval()


def count_params(models):
    return sum(p.numel() for model in models for p in model.parameters())


def tile_starts(length: int, tile: int, overlap: int):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, tile - overlap))
    starts.append(length - tile)
    return starts


def predict_tiled(predict, inputs, tile: int, overlap: int):
    """Run one image in overlapping tiles and stitch cropped tile centers."""
    hr = inputs[0] if isinstance(inputs, tuple) else inputs
    h, w = hr.shape[-2:]
    ys, xs = tile_starts(h, tile, overlap), tile_starts(w, tile, overlap)
    output = weights = None
    for yi, y0 in enumerate(ys):
        for xi, x0 in enumerate(xs):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            patch_hr = hr[..., y0:y1, x0:x1]
            if isinstance(inputs, tuple):
                scale = inputs[1].shape[-2] / h
                ly0, lx0 = int(round(y0 * scale)), int(round(x0 * scale))
                ly1, lx1 = int(round(y1 * scale)), int(round(x1 * scale))
                patch_input = (patch_hr, inputs[1][..., ly0:ly1, lx0:lx1])
            else:
                patch_input = patch_hr
            pred = predict(patch_input)
            if output is None:
                output = pred.new_zeros((pred.shape[0], pred.shape[1], h, w))
                weights = pred.new_zeros((1, 1, h, w))
            top = 0 if yi == 0 else overlap // 2
            left = 0 if xi == 0 else overlap // 2
            bottom = pred.shape[-2] if yi == len(ys) - 1 else pred.shape[-2] - (overlap - overlap // 2)
            right = pred.shape[-1] if xi == len(xs) - 1 else pred.shape[-1] - (overlap - overlap // 2)
            gy0, gx0, gy1, gx1 = y0 + top, x0 + left, y0 + bottom, x0 + right
            output[..., gy0:gy1, gx0:gx1] += pred[..., top:bottom, left:right]
            weights[..., gy0:gy1, gx0:gx1] += 1
    return output / weights.clamp_min(1)


def exp7c(device):
    metrics = import_file(EDGE / "evaluate_exp7c_metrics.py", "bench_exp7c_metrics")
    definitions = metrics.load_model_definitions(EDGE / "experiments/exp7C_joint_fidelity_20260818.py")
    root = EDGE / "experiments/exp7/exp7C_20260818_joint_fidelity/models/exp7C_20260818_joint_fidelity"
    base = definitions["DeepGuidedFilter"](radius=1, eps=1e-8).to(device)
    refine = definitions["CascadeRefine3Stage"](base_channels=32, a_scale=0.2, b_scale=0.2,
                                                 use_delta_as_input=True).to(device)
    metrics.load_state(base, root / "convguided_joint_best_20260818_l0p500_a1p400.pth", device)
    metrics.load_state(refine, root / "rf_3stage_best_20260818_l0p500_a1p400.pth", device)
    return [base, refine], lambda inputs: (refine(inputs[0] - 0.5,
        base(inputs[1] - 0.5, inputs[0] - 0.5) - 0.5)["final"] + 0.5).clamp(0, 1), {
        "base": str(root / "convguided_joint_best_20260818_l0p500_a1p400.pth"),
        "refine": str(root / "rf_3stage_best_20260818_l0p500_a1p400.pth")}


def build_candidate(task: str, name: str, device):
    if task == "edge" and name == "edge_exp7c":
        return name, *exp7c(device)
    if task == "edge" and name == "edge_nafnet":
        mod = import_file(EDGE / "baseline/evaluate_nafnet_wls.py", "bench_edge_naf")
        path = EDGE / "baseline/runs/nafnet_wls_patch256_nccl_3gpu_bs32/nafnet_wls_best.pth"
        model = load_state(mod.build_model(device), path, "model_state")
        return name, [model], lambda inputs: model(inputs[0]), {"model": str(path)}
    if task == "edge" and name == "edge_vivnet":
        from baseline.vivnet_wls.model import build_model, run_padded
        path = EDGE / "baseline/runs/vivnet_wls_patch128_nccl_4gpu_bs48/vivnet_wls_best.pth"
        model = load_state(build_model(device), path, "model_state")
        return name, [model], lambda inputs: run_padded(model, inputs[0]), {"model": str(path)}
    if task == "clipart" and name == "clipart_exp3b":
        ev = import_file(CLIPART / "evaluate_all_models.py", "bench_clipart_evaluation")
        ours = ev.load_ours("exp3-b", CLIPART, device)
        def predict(inputs):
            x, lr = inputs
            coarse = ours["base"](lr - 0.5, x - 0.5)
            return (ours["refine"](x - 0.5, coarse - 0.5)["final"] + 0.5).clamp(0, 1)
        return name, [ours["base"], ours["refine"]], predict, {"model": str(ours["checkpoint"])}
    if task == "clipart" and name == "clipart_vivnet":
        ev = import_file(CLIPART / "evaluate_all_models.py", "bench_clipart_evaluation")
        model = ev.load_baselines(CLIPART, device, ["VIVNet"])["VIVNet"]
        def predict(inputs):
            x = inputs[0]
            h, w = x.shape[-2:]
            padded = F.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode="reflect")
            return model(padded)[..., :h, :w]
        return name, [model], predict, {"model": ev.BASELINE_CHECKPOINTS["VIVNet"]}
    if task == "texture" and name.startswith("texture_"):
        model_name = name.removeprefix("texture_")
        sys.path.insert(0, str(TEXTURE))
        baseline_package = sys.modules.get("baseline")
        if baseline_package is not None:
            baseline_package.__path__.insert(0, str(TEXTURE / "baseline"))
        ev = import_file(TEXTURE / "eval_rtv_models.py", "bench_texture_evaluation")
        checkpoint = ev.DEFAULT_CHECKPOINTS[model_name]
        predict, modules = ev.load_model(model_name, ev.DEFAULT_CHECKPOINTS,
                                         ev.DEFAULT_OFFICIAL_SCRIPT, device)
        return name, modules, predict, {"model": str(checkpoint)}
    raise ValueError((task, name))


def read_source(task: str):
    if task == "edge":
        path = sorted((EDGE.parent.parent.parent / "S2R-HDR/data/20260818/testinput-l0p500-a1p400-200").glob("*.tiff"))[0]
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        return path, image[None, None]
    if task == "clipart":
        path = sorted((CLIPART / "openclipart_5000/jpeg_pairs_512/jpeg_q30").glob("*.jpg"))[0]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return path, rgb.transpose(2, 0, 1)[None]
    path = sorted((TEXTURE.parent.parent / "datasets/VOC2012/RTV_RGB_fixed_lambda_0p01/test/input").glob("*.tiff"))[0]
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    if bgr.max() > 1:
        bgr /= 255.0 if bgr.max() <= 255 else 65535.0
    return path, bgr.transpose(2, 0, 1)[None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("edge", "clipart", "texture", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--output", type=Path, default=HERE / "results/4k_task_benchmarks.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run through the established WinTrain environment")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    task_names = ("edge", "clipart", "texture") if args.task == "all" else (args.task,)
    results = []

    def save_report():
        report = {"protocol": {"resolution": [args.width, args.height], "batch_size": 1,
            "precision": "fp32", "device": str(device), "gpu": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "warmup": args.warmup,
            "repeats": args.repeats, "timing": "CUDA events, GPU-resident input; tiled forward only",
            "tiling": {"tile_size": args.tile_size, "overlap": args.tile_overlap},
            "resize_test": "representative native task test images resized to 3840x2160; synthetic 4K stress test",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, "results": results}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for task in task_names:
        source_path, source = read_source(task)
        mode = "bilinear" if task == "edge" else "bicubic"
        image = F.interpolate(torch.from_numpy(source), size=(args.height, args.width),
                              mode=mode, align_corners=False).clamp(0, 1).to(device)
        lr = F.interpolate(image, size=(args.height // 2, args.width // 2),
                           mode="bilinear", align_corners=False)
        model_input = (image, lr) if task in {"edge", "clipart"} else image
        names = {"edge": ("edge_exp7c", "edge_nafnet", "edge_vivnet"),
                 "clipart": ("clipart_exp3b", "clipart_vivnet"),
                 "texture": ("texture_stage3", "texture_official", "texture_nafnet_patch")}[task]
        for name in names:
            name, modules, predict, ckpt = build_candidate(task, name, device)
            params = count_params(modules)
            record = {"task": task, "model": name, "parameters": params,
                      "parameters_million": params / 1e6, "checkpoints": ckpt,
                      "source_image": str(source_path), "input_shape": list(image.shape),
                      "status": "ok"}
            try:
                with torch.inference_mode():
                    for _ in range(args.warmup):
                        predict_tiled(predict, model_input, args.tile_size, args.tile_overlap)
                    torch.cuda.synchronize(device)
                    samples = []
                    for _ in range(args.repeats):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        predict_tiled(predict, model_input, args.tile_size, args.tile_overlap)
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end))
                record.update({"latency_ms_median": statistics.median(samples),
                               "latency_ms_samples": samples})
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                record.update({"status": "oom", "error": str(exc).splitlines()[0]})
            print(json.dumps(record), flush=True)
            results.append(record)
            save_report()
            del modules, predict
            torch.cuda.empty_cache()
    save_report()
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
