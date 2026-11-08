"""Save NAFNet, VIVNet, and exp7C predictions for test_fei visualization."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


EXP_A_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXP_A_ROOT.parent
EDGE_ROOT = PROJECT_ROOT.parent / "20260815_edgepreserving"
DEFAULT_DATA_ROOT = Path("/mnt/d/ResearchCode/S2R-HDR/data/test_fei/generated_20260818")
DEFAULT_OUTPUT_ROOT = EXP_A_ROOT / "evaluation" / "test_fei_modelvis"
DEFAULT_NAFNET_CHECKPOINT = (
    EDGE_ROOT / "baseline/runs/nafnet_wls_patch256_nccl_3gpu_bs32/nafnet_wls_best.pth"
)
DEFAULT_VIVNET_CHECKPOINT = (
    EDGE_ROOT / "baseline/runs/vivnet_wls_patch128_nccl_4gpu_bs48/vivnet_wls_best.pth"
)
DEFAULT_EXP7C_ROOT = (
    EDGE_ROOT / "experiments/exp7/exp7C_20260818_joint_fidelity/models/exp7C_20260818_joint_fidelity"
)
EXP7C_SOURCE = EDGE_ROOT / "experiments/exp7C_joint_fidelity_20260818.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--nafnet-checkpoint", type=Path, default=DEFAULT_NAFNET_CHECKPOINT)
    parser.add_argument("--vivnet-checkpoint", type=Path, default=DEFAULT_VIVNET_CHECKPOINT)
    parser.add_argument("--exp7c-root", type=Path, default=DEFAULT_EXP7C_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    return parser.parse_args()


def load_exp7c_definitions():
    source = EXP7C_SOURCE.read_text(encoding="utf-8")
    marker = "\ntrain_dataset = GuidedFilterDataset("
    if marker not in source:
        raise RuntimeError(f"Could not find safe model-definition boundary in {EXP7C_SOURCE}")
    namespace = {"__name__": "exp7c_test_fei_infer", "__file__": str(EXP7C_SOURCE)}
    exec(compile(source.split(marker, 1)[0], str(EXP7C_SOURCE), "exec"), namespace)
    return namespace


def load_state(model, checkpoint_path: Path, device: torch.device, state_key: str | None = None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint[state_key] if state_key else checkpoint
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return checkpoint


def even_tensor(image: torch.Tensor) -> torch.Tensor:
    height, width = image.shape[-2:]
    return image[..., : height - height % 2, : width - width % 2]


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2:
        raise ValueError(f"Expected a readable single-channel TIFF: {path}")
    image = image.astype(np.float32, copy=False)
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite pixels in {path}")
    return image


def save_matlab_tiff(path: Path, image: torch.Tensor) -> None:
    array = image.detach().float().squeeze().cpu().numpy()
    array = np.rint(np.clip(array, 0.0, 1.0) * 65535.0).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), array):
        raise IOError(f"Could not save prediction TIFF: {path}")


def autocast(device: torch.device, precision: str):
    enabled = precision == "bf16" and device.type == "cuda"
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu explicitly")
    device = torch.device(args.device)

    input_dir = args.data_root / "testinput-l0p500-a1p400"
    lr_dir = args.data_root / "testinput-l0p500-a1p400_downsample"
    label_dir = args.data_root / "testlabel-l0p500-a1p400"
    for directory in (input_dir, lr_dir, label_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing 20260818 data directory: {directory}")
    inputs = sorted(input_dir.glob("*.tif*"))
    if not inputs:
        raise RuntimeError(f"No input TIFFs found in {input_dir}")

    sys.path.insert(0, str(EDGE_ROOT))
    sys.path.insert(0, str(EDGE_ROOT / "baseline" / "NAFNet"))
    from basicsr.models.archs.NAFNet_arch import NAFNetLocal
    from baseline.vivnet_wls.model import build_model as build_vivnet
    from baseline.vivnet_wls.model import run_padded as run_vivnet_padded

    nafnet = NAFNetLocal(
        img_channel=1,
        width=32,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
        train_size=(1, 1, 256, 256),
    ).to(device)
    load_state(nafnet, args.nafnet_checkpoint, device, "model_state")
    nafnet.eval()

    vivnet_checkpoint = torch.load(args.vivnet_checkpoint, map_location="cpu", weights_only=False)
    if vivnet_checkpoint.get("run_protocol", {}).get("inp_channels") != 1:
        raise ValueError("VIVNet checkpoint is not the single-channel WLS model")
    vivnet = build_vivnet(device)
    vivnet.load_state_dict(vivnet_checkpoint["model_state"], strict=True)
    vivnet.eval()

    definitions = load_exp7c_definitions()
    exp7c_conv = definitions["DeepGuidedFilter"](radius=1, eps=1e-8).to(device)
    exp7c_refine = definitions["CascadeRefine3Stage"](
        base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True
    ).to(device)
    load_state(
        exp7c_conv,
        args.exp7c_root / "convguided_joint_best_20260818_l0p500_a1p400.pth",
        device,
    )
    load_state(
        exp7c_refine,
        args.exp7c_root / "rf_3stage_best_20260818_l0p500_a1p400.pth",
        device,
    )
    exp7c_conv.eval()
    exp7c_refine.eval()

    param_tag = "wls_l0p500_a1p400"
    model_dirs = {
        name: args.output_root / param_tag / name
        for name in ("nafnet", "vivnet", "exp7c_recipe")
    }
    for directory in model_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.inference_mode():
        for index, input_path in enumerate(inputs, 1):
            scene = input_path.stem.removesuffix("-log")
            lr_path = lr_dir / input_path.name
            label_path = label_dir / f"{scene}-WLS-l0p500-a1p400.tiff"
            if not lr_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(f"Missing paired LR/label for scene {scene}")

            input_np = read_gray(input_path)
            label_np = read_gray(label_path)
            input_tensor = torch.from_numpy(input_np.copy())[None, None].to(device)
            label_tensor = torch.from_numpy(label_np.copy())[None, None].to(device)

            with autocast(device, args.precision):
                naf_prediction = nafnet(input_tensor)
                viv_prediction = run_vivnet_padded(vivnet, input_tensor)

            lr_tensor = torch.from_numpy(read_gray(lr_path).copy())[None, None].to(device)
            exp7c_input = even_tensor(input_tensor)
            exp7c_label = even_tensor(label_tensor)
            with autocast(device, args.precision):
                coarse = exp7c_conv(lr_tensor - 0.5, exp7c_input - 0.5)
                exp7c_prediction = torch.clamp(
                    exp7c_refine(exp7c_input - 0.5, coarse - 0.5)["final"] + 0.5,
                    0.0,
                    1.0,
                )

            predictions = {
                "nafnet": naf_prediction.float().clamp(0.0, 1.0),
                "vivnet": viv_prediction.float().clamp(0.0, 1.0),
                "exp7c_recipe": exp7c_prediction.float().clamp(0.0, 1.0),
            }
            target = even_tensor(label_tensor)
            direct_predictions = (predictions["nafnet"], predictions["vivnet"])
            if any(prediction.shape[-2:] != input_tensor.shape[-2:] for prediction in direct_predictions):
                raise ValueError(f"A direct restoration model changed output size for {scene}")
            scene_row = {"scene": scene}
            for model_name, prediction in predictions.items():
                model_target = target if model_name == "exp7c_recipe" else label_tensor
                if prediction.shape != model_target.shape:
                    raise ValueError(
                        f"{model_name} output shape {tuple(prediction.shape)} does not match "
                        f"target {tuple(model_target.shape)} for {scene}"
                    )
                save_matlab_tiff(model_dirs[model_name] / f"{scene}.tiff", prediction)
                error = prediction.float() - model_target.float()
                mse = float(error.square().mean().cpu())
                scene_row[f"{model_name}_mse"] = mse
                scene_row[f"{model_name}_psnr_db"] = 10.0 * math.log10(1.0 / max(mse, 1e-12))
            rows.append(scene_row)
            print(f"[{index}/{len(inputs)}] saved {scene}", flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / param_tag / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "data_root": str(args.data_root.resolve()),
        "wls_parameters": {"lambda": 0.5, "alpha": 1.4},
        "images": len(rows),
        "precision": args.precision,
        "prediction_encoding": "single-channel uint16 TIFF normalized from [0,1]",
        "model_directories": {name: str(path.resolve()) for name, path in model_dirs.items()},
    }
    (args.output_root / param_tag / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
