#!/usr/bin/env python3
"""Run the exp8B mixed-label ConvGuided + 3-stage Refine cascade on clip-art JPEGs.

The exp7C checkpoints were trained with single-channel images. RGB JPEGs are
therefore converted to YCbCr: only the luminance Y channel is processed, while
the degraded input's Cb/Cr channels are retained for color visualization.
For each degraded JPEG:

    JPEG RGB -> Y channel HR input
                         + half-resolution Y LR input
                      -> ConvGuided checkpoint
                      -> 3-stage Refine checkpoint
                      -> predicted Y + original Cb/Cr -> color PNG

The clean PNG is never fed to the model and is optional. If it is absent, the
script still performs color inference and visualization without PSNR/MAE.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
EDGE_ROOT = PROJECT_ROOT.parent / "20260815_edgepreserving"
DEFAULT_DATA = PROJECT_ROOT / "openclipart_500" / "jpeg_pairs_512"
DEFAULT_CONV = (
    EDGE_ROOT
    / "experiments/exp8/exp8B_mixed_joint_fidelity/models/"
    "exp8B_mixed_joint_fidelity/convguided_joint_best_20260818_l0p500_a1p400.pth"
)
DEFAULT_RF = (
    EDGE_ROOT
    / "experiments/exp8/exp8B_mixed_joint_fidelity/models/"
    "exp8B_mixed_joint_fidelity/rf_3stage_best_20260818_l0p500_a1p400.pth"
)
# exp8B was trained on mixed_lavel_singlehdr. Its checkpoint suffix is a
# legacy name and must not be interpreted as a single WLS parameter pair.
RF_SOURCE = EDGE_ROOT / "experiments/exp8/exp8B_mixed_joint_fidelity.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_DATA / "jpeg_q30")
    p.add_argument("--clean-dir", type=Path, default=DEFAULT_DATA / "clean_png")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "exp8b_q30_10",
    )
    p.add_argument("--conv-checkpoint", type=Path, default=DEFAULT_CONV)
    p.add_argument("--rf-checkpoint", type=Path, default=DEFAULT_RF)
    p.add_argument("--rf-source", type=Path, default=RF_SOURCE)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-images", type=int, default=10)
    return p.parse_args()


def load_definitions(source_path: Path, name: str) -> ModuleType:
    """Load imports/classes/functions without executing training setup."""
    source = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    module = ModuleType(name)
    module.__file__ = str(source_path)
    allowed = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    nodes = [node for node in source.body if isinstance(node, allowed)]
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec")
    exec(code, module.__dict__)
    return module


def load_state(model, path: Path, torch) -> None:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def load_ycbcr(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        ycbcr = np.asarray(image.convert("YCbCr"), dtype=np.float32) / 255.0
    return ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]


def downsample_half(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    h -= h % 2
    w -= w % 2
    image = image[:h, :w]
    if h >= w:
        new_h = max(2, h // 2)
        new_w = max(2, round(w * new_h / h))
    else:
        new_w = max(2, w // 2)
        new_h = max(2, round(h * new_w / w))
    resized = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return np.asarray(resized.resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    return 100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)


def main() -> int:
    args = parse_args()
    for path in [args.input_dir, args.conv_checkpoint, args.rf_checkpoint, args.rf_source]:
        if not path.exists():
            raise SystemExit(f"Missing path: {path}")
    clean_available = args.clean_dir.exists()

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    conv_source = load_definitions(args.rf_source, "exp8b_definitions")
    conv = conv_source.DeepGuidedFilter(radius=1, eps=1e-8).to(device)
    rf = conv_source.CascadeRefine3Stage(
        base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True
    ).to(device)
    load_state(conv, args.conv_checkpoint, torch)
    load_state(rf, args.rf_checkpoint, torch)
    conv.eval()
    rf.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.jpg"))
    if args.max_images is not None:
        files = files[: args.max_images]
    if not files:
        raise SystemExit(f"No JPG files found under {args.input_dir}")

    rows = []
    with torch.no_grad():
        for index, jpeg_path in enumerate(files):
            clean_path = args.clean_dir / f"{jpeg_path.stem}.png"
            has_clean = clean_available and clean_path.is_file()
            if not has_clean:
                print(f"[skip] missing clean GT: {clean_path.name}")
                continue

            y_np, cb_np, cr_np = load_ycbcr(jpeg_path)
            hr_np = y_np
            lr_np = downsample_half(hr_np)

            lr = torch.from_numpy(lr_np)[None, None].to(device)
            hr = torch.from_numpy(hr_np)[None, None].to(device)
            coarse = conv(lr - 0.5, hr - 0.5)
            final = torch.clamp(rf(hr - 0.5, coarse - 0.5)["final"] + 0.5, 0, 1)
            output_y = final[0, 0].cpu().numpy()

            # Preserve the degraded input's chroma so the single-channel
            # exp7C prediction can be visualized as a color image.
            output_y_img = Image.fromarray(
                (output_y * 255.0).round().clip(0, 255).astype(np.uint8), mode="L"
            )
            cb_img = Image.fromarray((cb_np * 255.0).round().clip(0, 255).astype(np.uint8), mode="L")
            cr_img = Image.fromarray((cr_np * 255.0).round().clip(0, 255).astype(np.uint8), mode="L")
            output_color = Image.merge("YCbCr", (output_y_img, cb_img, cr_img)).convert("RGB")

            # Keep the three visual counterparts in the same folder:
            # original degraded input, clean GT, and actual model output.
            input_output_path = args.output_dir / f"{jpeg_path.stem}_input.jpg"
            clean_output_path = args.output_dir / f"{jpeg_path.stem}_clean_gt.png"
            output_path = args.output_dir / f"{jpeg_path.stem}_exp8b_output.png"
            with Image.open(jpeg_path) as original:
                original.convert("RGB").save(input_output_path, format="JPEG", quality=95)
            with Image.open(clean_path) as clean_image:
                clean_image.convert("RGB").save(clean_output_path, format="PNG")
            output_color.save(output_path, format="PNG")

            clean_y, _, _ = load_ycbcr(clean_path)
            row = {
                "input": input_output_path.name,
                "clean_gt": clean_output_path.name,
                "model_output": output_path.name,
                "input_psnr_y": psnr(hr_np, clean_y),
                "output_psnr_y": psnr(output_y, clean_y),
                "input_mae_y": float(np.mean(np.abs(hr_np - clean_y))),
                "output_mae_y": float(np.mean(np.abs(output_y - clean_y))),
            }
            rows.append(row)
            print(f"[{index + 1:04d}/{len(files)}] {jpeg_path.name}")

    if not rows:
        raise SystemExit("No samples were evaluated.")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metric_rows = [r for r in rows if "output_psnr_y" in r]
    summary = f"samples: {len(rows)}\n"
    if metric_rows:
        summary += (
            f"metric samples with clean reference: {len(metric_rows)}\n"
            f"mean input PSNR (Y): {np.mean([r['input_psnr_y'] for r in metric_rows]):.4f} dB\n"
            f"mean output PSNR (Y): {np.mean([r['output_psnr_y'] for r in metric_rows]):.4f} dB\n"
            f"mean input MAE (Y): {np.mean([r['input_mae_y'] for r in metric_rows]):.8f}\n"
            f"mean output MAE (Y): {np.mean([r['output_mae_y'] for r in metric_rows]):.8f}\n"
        )
    else:
        summary += "No clean reference found; visualization-only inference.\n"
    (args.output_dir / "metrics.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
