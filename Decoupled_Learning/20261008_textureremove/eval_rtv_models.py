"""Evaluate completed RTV models on the same RGB test images and geometry.

Timing covers model-specific tensor preprocessing, forward, and output scaling
on the selected device. It excludes image I/O, host-to-device transfer, metrics,
PNG writing, and checkpoint loading. CUDA is synchronized at both boundaries.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from baseline.official_model import load_official_model
from baseline.rtv_data import build_model_input
from rtv_training_common import (
    CascadeRefine3Stage,
    DeepGuidedFilter,
    make_low_resolution,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent.parent / "datasets" / "VOC2012" / "RTV_RGB_fixed_lambda_0p01"
DEFAULT_OFFICIAL_SCRIPT = ROOT.parents[2] / "decoupled-learning-fan" / "DecoupleLearning-master" / "train_1_operator.py"
MODEL_NAMES = ("stage1", "stage2", "stage3", "official", "nafnet_patch")
DEFAULT_CHECKPOINTS = {
    "stage1": ROOT / "models/conv_base_rtv/conv_base_rtv_best.pth",
    "stage2": ROOT / "models/refine_rtv/refine_rtv_best.pth",
    "stage3": ROOT / "models/joint_rtv/joint_rtv_best.pth",
    "official": ROOT / "baseline/runs/rtv_full_image/rtv_official_best.pth",
    "nafnet_patch": ROOT / "baseline/runs/nafnet_rtv_patch256_3gpu/nafnet_rtv_best.pth",
}


def discover_pairs(test_dir):
    """Return sorted test pairs; fail if either side has missing files."""
    test_dir = Path(test_dir)
    suffixes = {".tif", ".tiff"}
    inputs = {p.stem: p for p in (test_dir / "input").iterdir() if p.suffix.lower() in suffixes}
    labels = {p.stem: p for p in (test_dir / "label").iterdir() if p.suffix.lower() in suffixes}
    if not inputs or inputs.keys() != labels.keys():
        raise ValueError(
            f"Unmatched or empty input/label TIFF sets in {test_dir}: "
            f"inputs={len(inputs)}, labels={len(labels)}, "
            f"missing_labels={sorted(inputs.keys() - labels.keys())[:5]}, "
            f"missing_inputs={sorted(labels.keys() - inputs.keys())[:5]}"
        )
    return [(stem, inputs[stem], labels[stem]) for stem in sorted(inputs)]


def _read_bgr_float(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a three-channel image: {path}")
    image = image.astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError(f"Nonfinite pixels in {path}")
    if image.max() > 1.0:
        image /= 255.0 if image.max() <= 255.0 else 65535.0
    return np.clip(image, 0.0, 1.0)


def load_pair(input_path, label_path):
    """Use the stage loader's BGR convention and even top-left test crop."""
    image = _read_bgr_float(input_path)
    target = _read_bgr_float(label_path)
    height = min(image.shape[0], target.shape[0])
    width = min(image.shape[1], target.shape[1])
    height -= height % 2
    width -= width % 2
    if height < 2 or width < 2:
        raise ValueError(f"Image too small after even trimming: {input_path}")
    image = torch.from_numpy(np.ascontiguousarray(image[:height, :width].transpose(2, 0, 1)))[None]
    target = torch.from_numpy(np.ascontiguousarray(target[:height, :width].transpose(2, 0, 1)))[None]
    return image, target


def score_prediction(prediction, target):
    """Per-image MSE and MAE in the shared [0,1] value range."""
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    error = prediction.float().clamp(0.0, 1.0) - target.float()
    return error.square().mean().item(), error.abs().mean().item()


def summarize_rows(rows):
    """Average per-image metrics; timing quantiles use image-level means."""
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    times = np.array([row["infer_ms"] for row in rows], dtype=np.float64)
    mses = np.array([row["mse"] for row in rows], dtype=np.float64)
    maes = np.array([row["mae"] for row in rows], dtype=np.float64)
    psnrs = [(-10.0 * math.log10(mse)) for mse in mses if mse > 0]
    return {
        "images": len(rows),
        "mse": float(mses.mean()),
        "mae": float(maes.mean()),
        "mean_psnr_db": float(np.mean(psnrs)) if len(psnrs) == len(rows) else None,
        "mean_infer_ms": float(times.mean()),
        "p50_infer_ms": float(np.percentile(times, 50)),
        "p90_infer_ms": float(np.percentile(times, 90)),
        "total_infer_s": float(times.sum() / 1000.0),
    }


def _load_state(model, state):
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval()


def load_model(name, checkpoints, official_script, device):
    """Load one model and return its full inference path on BGR [0,1]."""
    checkpoint = checkpoints[name]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing {name} checkpoint: {checkpoint}")

    if name in ("stage1", "stage2", "stage3"):
        base = DeepGuidedFilter().to(device)
        if name == "stage3":
            joint = torch.load(checkpoint, map_location="cpu", weights_only=False)
            base = _load_state(base, joint["base"])
            refine = _load_state(CascadeRefine3Stage().to(device), joint["refine"])
        else:
            base_state = torch.load(checkpoints["stage1"], map_location="cpu", weights_only=False)
            base = _load_state(base, base_state)
            if name == "stage2":
                refine_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                refine = _load_state(CascadeRefine3Stage().to(device), refine_state)

        def predict(image):
            centered = image - 0.5
            coarse = base(make_low_resolution(centered), centered)
            if name == "stage1":
                return coarse.clamp(0.0, 1.0)
            return refine(centered, coarse)["final"].clamp(0.0, 1.0)

        return predict, [base] + ([] if name == "stage1" else [refine])

    if name == "official":
        if not official_script.is_file():
            raise FileNotFoundError(f"Missing official source script: {official_script}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = _load_state(load_official_model(official_script)().to(device), state["model_state"])
        condition_value = state["rtv_condition"]

        def predict(image):
            image_255 = image * 255.0
            condition = torch.full((image.shape[0], 1), condition_value, device=image.device)
            return (model([condition, build_model_input(image_255)]) / 255.0).clamp(0.0, 1.0)

        return predict, [model]

    if name == "nafnet_patch":
        from baseline.train_nafnet_rtv import make_model_and_loss

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        protocol = state.get("run_protocol", {})
        if protocol.get("train_geometry") != "random_paired_patch" or protocol.get("train_crop_size") != 256:
            raise ValueError(f"Expected a completed patch-256 NAFNet checkpoint: {checkpoint}")
        if state.get("iter") != protocol.get("total_iter"):
            raise ValueError(f"NAFNet checkpoint is incomplete: {state.get('iter')} / {protocol.get('total_iter')}")
        model, _ = make_model_and_loss(device)
        model = _load_state(model, state["model_state"])
        return lambda image: model(image).clamp(0.0, 1.0), [model]

    raise ValueError(f"Unknown model: {name}")


def timed_prediction(predict, image, device, repeats):
    """Synchronize around the on-device inference path; return the final output."""
    elapsed = []
    output = None
    with torch.inference_mode():
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = predict(image)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed.append((time.perf_counter() - start) * 1000.0)
    return output, float(np.median(elapsed))


def _save_png(path, tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = tensor[0].detach().float().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    if not cv2.imwrite(str(path), np.rint(bgr * 255.0).astype(np.uint8)):
        raise IOError(f"Failed to save {path}")


def write_preview(output_dir, stem, models):
    """Write one labeled row of the input, RTV target, and model predictions."""
    output_dir = Path(output_dir)
    panels = []
    for name in ("input", "target", *models):
        path = output_dir / "images" / name / f"{stem}.png"
        panel = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if panel is None:
            raise FileNotFoundError(f"Missing preview panel: {path}")
        panel = cv2.copyMakeBorder(panel, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(panel, name, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(panel)
    path = output_dir / "comparisons" / f"{stem}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.hconcat(panels)):
        raise IOError(f"Failed to save preview: {path}")
    return path


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_empty_output(output_dir):
    """Keep separate runs from silently mixing CSVs and display images."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}; choose a new --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval_results/unified_rtv")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--warmup", type=int, default=2, help="Untimed runs on the first image per model")
    parser.add_argument("--repeats", type=int, default=1, help="Timed runs per image; report their median")
    parser.add_argument("--max-images", type=int, default=0, help="First N sorted pairs; 0 means all")
    parser.add_argument("--no-save-images", action="store_true", help="Skip display PNGs")
    parser.add_argument("--preview-count", type=int, default=16, help="Side-by-side previews from the first N pairs")
    parser.add_argument("--official-script", type=Path, default=DEFAULT_OFFICIAL_SCRIPT)
    for name, default in DEFAULT_CHECKPOINTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}-checkpoint", type=Path, default=default)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.warmup < 0 or args.repeats < 1 or args.max_images < 0 or args.preview_count < 0:
        raise ValueError("warmup, max-images, and preview-count must be nonnegative; repeats must be positive")
    if len(set(args.models)) != len(args.models):
        raise ValueError("Duplicate model names in --models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu or fix the CUDA environment")
    checkpoints = {name: getattr(args, f"{name}_checkpoint") for name in MODEL_NAMES}
    pairs = discover_pairs(args.data_root / "test")
    if args.max_images:
        pairs = pairs[:args.max_images]
    ensure_empty_output(args.output_dir)
    print(f"Test pairs: {len(pairs)}; device: {device}; models: {', '.join(args.models)}", flush=True)
    print("Timing: device tensor preprocessing + forward + output scaling; image I/O, transfers, metrics, and PNG saving excluded.", flush=True)
    summaries = []

    for name in args.models:
        print(f"Loading {name}: {checkpoints[name]}", flush=True)
        predict, modules = load_model(name, checkpoints, args.official_script, device)
        rows = []
        for index, (stem, input_path, label_path) in enumerate(pairs):
            image_cpu, target_cpu = load_pair(input_path, label_path)
            image = image_cpu.to(device)
            target = target_cpu.to(device)
            if index == 0:
                with torch.inference_mode():
                    for _ in range(args.warmup):
                        predict(image)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            prediction, infer_ms = timed_prediction(predict, image, device, args.repeats)
            mse, mae = score_prediction(prediction, target)
            rows.append({
                "image": stem,
                "height": image.shape[-2],
                "width": image.shape[-1],
                "mse": mse,
                "mae": mae,
                "psnr_db": -10.0 * math.log10(mse) if mse > 0 else "inf",
                "infer_ms": infer_ms,
            })
            if not args.no_save_images:
                _save_png(args.output_dir / "images" / name / f"{stem}.png", prediction)
                if name == args.models[0]:
                    _save_png(args.output_dir / "images/input" / f"{stem}.png", image_cpu)
                    _save_png(args.output_dir / "images/target" / f"{stem}.png", target_cpu)
            if (index + 1) % 50 == 0 or index + 1 == len(pairs):
                print(f"  {name}: {index + 1}/{len(pairs)}", flush=True)
        _write_csv(args.output_dir / f"{name}_per_image.csv", rows, list(rows[0]))
        summary = {"model": name, "checkpoint": str(checkpoints[name].resolve()), **summarize_rows(rows)}
        summaries.append(summary)
        print(f"  {name}: MSE={summary['mse']:.8f}, MAE={summary['mae']:.8f}, mean inference={summary['mean_infer_ms']:.2f} ms", flush=True)
        del predict, modules, prediction, image, target
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not args.no_save_images:
        for stem, _, _ in pairs[:args.preview_count]:
            write_preview(args.output_dir, stem, args.models)
    _write_csv(args.output_dir / "summary.csv", summaries, list(summaries[0]))
    manifest = {
        "data_root": str(args.data_root.resolve()),
        "test_images": len(pairs),
        "models": args.models,
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cpu_threads": torch.get_num_threads(),
        "precision": "fp32 for all models",
        "geometry": "shared full-image top-left crop to even height and width",
        "channel_order": "BGR for all models",
        "metric": "per-image pixel mean in [0,1], then mean across images; predictions clamped to [0,1]",
        "timing": "device tensor preprocessing + forward + output scaling and clamping; excludes disk I/O, host-to-device transfer, metrics, and image saving; CUDA synchronized; median of repeats per image",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "pngs_saved": not args.no_save_images,
        "comparison_previews": 0 if args.no_save_images else min(args.preview_count, len(pairs)),
        "checkpoint_paths": {name: str(checkpoints[name].resolve()) for name in args.models},
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Results: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
