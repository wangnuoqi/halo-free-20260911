"""Evaluate the exp7C joint ConvGuided + 3-stage model on the 20260818 test set."""

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/mnt/d/ResearchCode/S2R-HDR/data/20260818")
EXP_ROOT = ROOT / "experiments/exp7/exp7C_20260818_joint_fidelity/models/exp7C_20260818_joint_fidelity"
SOURCE = ROOT / "experiments/exp7C_joint_fidelity_20260818.py"


def load_model_definitions(source_path):
    # The source file contains a training bootstrap after this line. Execute
    # only the imports and class definitions so evaluation cannot start training.
    source = source_path.read_text(encoding="utf-8")
    # The original script creates its training dataset before the device and
    # training loop. Stop before that side effect, not merely before training.
    marker = "\ntrain_dataset = GuidedFilterDataset("
    if marker not in source:
        raise RuntimeError(f"Training bootstrap marker not found in {source_path}")
    namespace = {"__name__": "exp7c_eval_source", "__file__": str(source_path)}
    exec(compile(source.split(marker, 1)[0], str(source_path), "exec"), namespace)
    return namespace


def load_state(model, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def even(array):
    return array[: array.shape[-2] - array.shape[-2] % 2,
                 : array.shape[-1] - array.shape[-1] % 2]


def files(directory):
    return sorted(path for path in Path(directory).iterdir() if path.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=EXP_ROOT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "baseline/evaluation/exp7c_test")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    data_root = args.data_root
    lr_files = files(data_root / "testinput-l0p500-a1p400-200_downsample")
    origin_files = files(data_root / "testinput-l0p500-a1p400-200")
    label_files = files(data_root / "testlabel-l0p500-a1p400-200")
    if not lr_files or not (len(lr_files) == len(origin_files) == len(label_files)):
        raise ValueError(f"Mismatched test files: LR={len(lr_files)}, origin={len(origin_files)}, label={len(label_files)}")

    source = load_model_definitions(args.source)
    conv = source["DeepGuidedFilter"](radius=1, eps=1e-8).to(device)
    refine = source["CascadeRefine3Stage"](
        base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True
    ).to(device)
    load_state(conv, args.experiment_root / "convguided_joint_best_20260818_l0p500_a1p400.pth", device)
    load_state(refine, args.experiment_root / "rf_3stage_best_20260818_l0p500_a1p400.pth", device)
    conv.eval()
    refine.eval()

    rows = []
    with torch.inference_mode():
        for index, (lr_path, origin_path, label_path) in enumerate(
                zip(lr_files, origin_files, label_files), 1):
            lr = cv2.imread(str(lr_path), cv2.IMREAD_UNCHANGED).astype("float32")
            origin = cv2.imread(str(origin_path), cv2.IMREAD_UNCHANGED).astype("float32")
            label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED).astype("float32")
            origin = even(origin)
            label = even(label)
            lr = torch.from_numpy(lr)[None, None].to(device)
            origin = torch.from_numpy(origin)[None, None].to(device)
            label = torch.from_numpy(label)[None, None].to(device)
            coarse = conv(lr - 0.5, origin - 0.5)
            prediction = torch.clamp(
                refine(origin - 0.5, coarse - 0.5)["final"] + 0.5, 0.0, 1.0
            )
            error = prediction - label
            mse = float(error.square().mean().cpu())
            psnr = float(10.0 * math.log10(1.0 / max(mse, 1e-12)))
            rows.append({"id": origin_path.stem, "mse": mse, "psnr_db": psnr})
            if index % 25 == 0 or index == len(lr_files):
                print(f"test {index}/{len(lr_files)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "mse", "psnr_db"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": "exp7C joint ConvGuided + 3-stage",
        "data_root": str(data_root.resolve()),
        "test_images": len(rows),
        "mse": float(np.mean([row["mse"] for row in rows])),
        "mean_psnr_db": float(np.mean([row["psnr_db"] for row in rows])),
        "predictions_saved": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
