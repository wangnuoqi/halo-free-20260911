"""Affine expressivity oracle for the exp7C edge-preserving cascade.

This deliberately separates a useful per-image affine fit from the degenerate
per-pixel two-parameter fit.  The latter is an upper bound, not a learnable
model, because one scalar equation cannot identify two scalar parameters.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path("/mnt/d/ResearchCode/S2R-HDR/data/20260818")
DEFAULT_EXP = ROOT / "experiments/exp7/exp7C_20260818_joint_fidelity/models/exp7C_20260818_joint_fidelity"
SOURCE = ROOT / "experiments/exp7C_joint_fidelity_20260818.py"


def load_defs(source_path):
    source = source_path.read_text(encoding="utf-8")
    marker = "\ntrain_dataset = GuidedFilterDataset("
    namespace = {"__name__": "exp7c_affine_oracle", "__file__": str(source_path)}
    exec(compile(source.split(marker, 1)[0], str(source_path), "exec"), namespace)
    return namespace


def load_state(model, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def even(x):
    return x[: x.shape[-2] - x.shape[-2] % 2, : x.shape[-1] - x.shape[-1] % 2]


def mse_psnr(pred, target):
    mse = float(np.mean((pred - target) ** 2))
    return mse, float(10.0 * math.log10(1.0 / max(mse, 1e-12)))


def affine_image_fit(x, y):
    """Fit one (a,b) pair for the whole image, then return prediction."""
    xx = x.reshape(-1).astype(np.float64)
    yy = y.reshape(-1).astype(np.float64)
    xm, ym = xx.mean(), yy.mean()
    var = np.mean((xx - xm) ** 2)
    a = 0.0 if var < 1e-12 else float(np.mean((xx - xm) * (yy - ym)) / var)
    b = float(ym - a * xm)
    return (a * x + b).astype(np.float32), a, b


def bounded_pixel_oracle(x, y, amin=0.8, amax=1.2, bmin=-0.2, bmax=0.2, steps=81):
    """Approximate the per-pixel box-constrained oracle with an a grid.

    For every candidate a, the optimal b is clip(y-a*x, bmin, bmax).  The
    result is an upper bound on the exact box-constrained optimum because the
    a grid is discrete; 81 points are sufficient for a diagnostic report.
    """
    best = np.full_like(x, np.inf, dtype=np.float32)
    best_pred = np.empty_like(x, dtype=np.float32)
    for a in np.linspace(amin, amax, steps, dtype=np.float32):
        b = np.clip(y - a * x, bmin, bmax)
        pred = a * x + b
        err = np.abs(pred - y)
        mask = err < best
        best[mask] = err[mask]
        best_pred[mask] = pred[mask]
    return best_pred


def reachable_interval_prediction(x, target, stages, amin=0.8, amax=1.2, bmin=-0.2, bmax=0.2):
    """Best point in the interval reachable by `stages` bounded affine updates."""
    lo = x.astype(np.float32).copy()
    hi = x.astype(np.float32).copy()
    for _ in range(stages):
        lo = np.minimum(amin * lo, amax * lo) + bmin
        hi = np.maximum(amin * hi, amax * hi) + bmax
    return np.clip(target, lo, hi).astype(np.float32), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--experiment-root", type=Path, default=DEFAULT_EXP)
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "baseline/evaluation/affine_oracle_exp7c")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ns = load_defs(args.source)
    conv = ns["DeepGuidedFilter"](radius=1, eps=1e-8).to(device)
    refine = ns["CascadeRefine3Stage"](base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True).to(device)
    load_state(conv, args.experiment_root / "convguided_joint_best_20260818_l0p500_a1p400.pth", device)
    load_state(refine, args.experiment_root / "rf_3stage_best_20260818_l0p500_a1p400.pth", device)
    conv.eval(); refine.eval()

    root = args.data_root
    lrs = sorted((root / "testinput-l0p500-a1p400-200_downsample").iterdir())
    origins = sorted((root / "testinput-l0p500-a1p400-200").iterdir())
    labels = sorted((root / "testlabel-l0p500-a1p400-200").iterdir())
    if not (len(lrs) == len(origins) == len(labels) == 200):
        raise ValueError(f"expected 200 aligned test images, got {len(lrs)}, {len(origins)}, {len(labels)}")

    names = ["input_origin", "coarse", "current_stage1", "current_stage2", "current_exp7c", "coarse_image_affine", "stage1_delta_affine",
             "stage2_delta_affine", "pixel_box_coarse", "pixel_free_coarse",
             "pixel_box_delta1", "pixel_box_delta2", "pixel_free_delta1",
             "pixel_free_delta2", "reachable_d1_2stages", "reachable_zero_3stages",
             "reachable_zero_2stages",
             "current_exp7c"]
    sums = {name: [] for name in names}
    rows = []
    with torch.inference_mode():
        for i, (lp, op, yp) in enumerate(zip(lrs, origins, labels), 1):
            lr = cv2.imread(str(lp), cv2.IMREAD_UNCHANGED).astype("float32")
            origin = even(cv2.imread(str(op), cv2.IMREAD_UNCHANGED).astype("float32"))
            gt = even(cv2.imread(str(yp), cv2.IMREAD_UNCHANGED).astype("float32"))
            lr_t = torch.from_numpy(lr)[None, None].to(device)
            origin_t = torch.from_numpy(origin)[None, None].to(device)
            coarse_t = conv(lr_t - 0.5, origin_t - 0.5)
            out = refine(origin_t - 0.5, coarse_t - 0.5)
            coarse = coarse_t[0, 0].cpu().numpy().astype("float32")
            d1 = out["delta1"][0, 0].cpu().numpy().astype("float32")
            d2 = out["delta2"][0, 0].cpu().numpy().astype("float32")
            current_stage1 = np.clip(out["coarse1"][0, 0].cpu().numpy() + 0.5, 0, 1).astype("float32")
            current_stage2 = np.clip(out["coarse2"][0, 0].cpu().numpy() + 0.5, 0, 1).astype("float32")
            current = np.clip(out["final"][0, 0].cpu().numpy() + 0.5, 0, 1).astype("float32")
            gt = np.clip(gt, 0, 1)
            coarse = np.clip(coarse, 0, 1)
            target_delta = gt - coarse

            image_aff, a_img, b_img = affine_image_fit(coarse, gt)
            d1_aff, a_d1, b_d1 = affine_image_fit(d1, target_delta)
            d2_aff, a_d2, b_d2 = affine_image_fit(d2, target_delta)
            stage1_aff = np.clip(coarse + d1_aff, 0, 1)
            stage2_aff = np.clip(coarse + d2_aff, 0, 1)
            box_coarse = bounded_pixel_oracle(coarse, gt)
            free_coarse = gt.copy()
            box_d1 = bounded_pixel_oracle(d1, target_delta)
            box_d2 = bounded_pixel_oracle(d2, target_delta)
            free_d1 = target_delta.copy()
            free_d2 = target_delta.copy()
            reachable_d1, _, _ = reachable_interval_prediction(d1, target_delta, stages=2)
            reachable_zero, _, _ = reachable_interval_prediction(np.zeros_like(target_delta), target_delta, stages=3)
            reachable_zero2, _, _ = reachable_interval_prediction(np.zeros_like(target_delta), target_delta, stages=2)
            preds = {"input_origin": origin, "coarse": coarse,
                     "current_stage1": current_stage1, "current_stage2": current_stage2,
                     "current_exp7c": current,
                     "coarse_image_affine": np.clip(image_aff, 0, 1),
                     "stage1_delta_affine": stage1_aff, "stage2_delta_affine": stage2_aff,
                     "pixel_box_coarse": box_coarse, "pixel_free_coarse": free_coarse,
                     "pixel_box_delta1": np.clip(coarse + box_d1, 0, 1),
                     "pixel_box_delta2": np.clip(coarse + box_d2, 0, 1),
                     "pixel_free_delta1": np.clip(coarse + free_d1, 0, 1),
                     "pixel_free_delta2": np.clip(coarse + free_d2, 0, 1),
                     "reachable_d1_2stages": np.clip(coarse + reachable_d1, 0, 1),
                     "reachable_zero_3stages": np.clip(coarse + reachable_zero, 0, 1),
                     "reachable_zero_2stages": np.clip(coarse + reachable_zero2, 0, 1)}
            metrics = {"id": op.stem, "a_image": a_img, "b_image": b_img,
                       "a_delta1": a_d1, "b_delta1": b_d1, "a_delta2": a_d2, "b_delta2": b_d2}
            for name, pred in preds.items():
                mse, psnr = mse_psnr(pred, gt)
                sums[name].append((mse, psnr))
                metrics[f"{name}_mse"] = mse
                metrics[f"{name}_psnr_db"] = psnr
            rows.append(metrics)
            if i % 25 == 0 or i == 200:
                print(f"test {i}/200", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary = {
        "protocol": "20260818 test 200, even crop, full image; affine oracle diagnostics",
        "warning": "pixel_free_coarse is a degenerate oracle and is not learnable; one pixel gives one equation for two parameters",
        "test_images": 200,
        "mean_of_per_image_mse": {k: float(np.mean([v[0] for v in vals])) for k, vals in sums.items()},
        "mean_of_per_image_psnr_db": {k: float(np.mean([v[1] for v in vals])) for k, vals in sums.items()},
        "pixel_box_bounds": {"a": [0.8, 1.2], "b": [-0.2, 0.2], "a_grid_steps": 81},
        "source_checkpoints": {
            "conv": str(args.experiment_root / "convguided_joint_best_20260818_l0p500_a1p400.pth"),
            "refine": str(args.experiment_root / "rf_3stage_best_20260818_l0p500_a1p400.pth"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
