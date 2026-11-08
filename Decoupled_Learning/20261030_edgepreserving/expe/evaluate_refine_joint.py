"""Evaluate Exp E base/refine/joint on centered 512x512 test crops."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

from exp_b.src.data_exp_b import PairedTripletDataset
from exp_b.src.evaluate_exp_b import image_metrics, report_crop_triplet
from exp_b.src.train_exp_b import forward_phase
from expe.refine_joint import FullExpE


def load_base(model: FullExpE, path: Path, variant: str) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol", {}).get("variant") != variant:
        raise ValueError(f"expected {variant} Base checkpoint: {path}")
    model.base.network.load_state_dict(checkpoint["model"], strict=True)


def load_full(model: FullExpE, path: Path, phase: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("phase") != phase:
        raise ValueError(f"{path} is phase={checkpoint.get('phase')!r}, expected {phase!r}")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return checkpoint


def load_legacy_exp7c(source_path: Path):
    source = source_path.read_text(encoding="utf-8")
    marker = "\ntrain_dataset = GuidedFilterDataset("
    if marker not in source:
        raise RuntimeError(f"could not find model-only boundary in {source_path}")
    namespace = {"__name__": "exp7c_512_evaluator", "__file__": str(source_path)}
    exec(compile(source.split(marker, 1)[0], str(source_path), "exec"), namespace)
    return namespace


def load_legacy_state(model, path: Path, device: torch.device) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/d/ResearchCode/S2R-HDR/data/20260818"))
    parser.add_argument("--variant", default="viv_gf_e1")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--refine-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--legacy-exp7c-conv-checkpoint", type=Path, required=True)
    parser.add_argument("--legacy-exp7c-refine-checkpoint", type=Path, required=True)
    parser.add_argument("--legacy-exp7c-source", type=Path,
                        default=Path("../20260815_edgepreserving/experiments/exp7C_joint_fidelity_20260818.py"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = PairedTripletDataset(args.data_root, split="test")
    models = {}
    base_model = FullExpE(args.variant)
    load_base(base_model, args.base_checkpoint, args.variant)
    models["base"] = base_model.to(device).eval()
    phase_checkpoints = {}
    for phase, path in (("refine", args.refine_checkpoint), ("joint", args.joint_checkpoint)):
        model = FullExpE(args.variant)
        phase_checkpoints[phase] = load_full(model, path, phase)
        models[phase] = model.to(device).eval()
    legacy_source = load_legacy_exp7c(args.legacy_exp7c_source)
    legacy_base = legacy_source["DeepGuidedFilter"](radius=1, eps=1e-8)
    legacy_refine = legacy_source["CascadeRefine3Stage"](
        base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True
    )
    load_legacy_state(legacy_base, args.legacy_exp7c_conv_checkpoint, device)
    load_legacy_state(legacy_refine, args.legacy_exp7c_refine_checkpoint, device)

    rows = []
    with torch.inference_mode():
        for index, (lr, hr, gt) in enumerate(dataset):
            lr, hr, gt = report_crop_triplet(lr.numpy(), hr.numpy(), gt.numpy())
            lr = torch.from_numpy(np.ascontiguousarray(lr))[None].to(device)
            hr = torch.from_numpy(np.ascontiguousarray(hr))[None].to(device)
            gt = torch.from_numpy(np.ascontiguousarray(gt))[None].to(device)
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if args.precision == "bf16" and device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            with autocast:
                predictions = {
                    "base_coarse": models["base"].base(lr - 0.5, hr - 0.5),
                }
                legacy_coarse = legacy_base(lr - 0.5, hr - 0.5)
                legacy_outputs = legacy_refine(hr - 0.5, legacy_coarse - 0.5)
                predictions["legacy_exp7c_coarse"] = legacy_coarse
                predictions["legacy_exp7c_final"] = (legacy_outputs["final"] + 0.5).clamp(0.0, 1.0)
                for phase in ("refine", "joint"):
                    outputs = models[phase](lr - 0.5, hr - 0.5)
                    predictions[f"{phase}_coarse"] = outputs["coarse0"]
                    predictions[f"{phase}_final"] = outputs["final"]
            target = gt[0, 0].cpu().numpy()
            for name, prediction in predictions.items():
                image = prediction.float().clamp(0.0, 1.0)[0, 0].cpu().numpy()
                rows.append({"image_id": dataset.ids[index], "output": name,
                             **image_metrics(image, target)})
            if (index + 1) % 25 == 0 or index + 1 == len(dataset):
                print(f"test-200 512 crop: {index + 1}/{len(dataset)}", flush=True)

    summary_rows = []
    for output in sorted({row["output"] for row in rows}):
        group = [row for row in rows if row["output"] == output]
        mean_mse = float(np.mean([row["mse"] for row in group]))
        summary_rows.append({
            "output": output, "images": len(group), "mean_image_mse": mean_mse,
            "mean_per_image_psnr_db": float(np.mean([row["psnr_db"] for row in group])),
            "psnr_from_mean_mse_db": 10.0 * math.log10(1.0 / max(mean_mse, 1e-12)),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment": f"Exp E {args.variant} Base + three-stage Refine + Joint",
        "variant": args.variant,
        "data_root": str(args.data_root.resolve()), "test_images": len(dataset),
        "geometry": "centered_512x512_hr_crop_aligned_256x256_lr",
        "odd_border_policy": "trim_bottom_right_before_center_crop",
        "precision": args.precision, "device": str(device),
        "checkpoints": {
            "base": str(args.base_checkpoint.resolve()),
            "refine": str(args.refine_checkpoint.resolve()),
            "joint": str(args.joint_checkpoint.resolve()),
            "legacy_exp7c_conv": str(args.legacy_exp7c_conv_checkpoint.resolve()),
            "legacy_exp7c_refine": str(args.legacy_exp7c_refine_checkpoint.resolve()),
        },
        "results": summary_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
