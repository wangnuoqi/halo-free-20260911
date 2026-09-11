"""Evaluate the trained ConvGuided and RF models without entering training."""

import argparse
import importlib.util
import math
import os
from pathlib import Path


def build_test_paths(data_root, experiment_root):
    data_root = Path(data_root)
    experiment_root = Path(experiment_root)
    return {
        "lr": data_root / "testinput-l0p500-a1p400-200_downsample",
        "origin": data_root / "testinput-l0p500-a1p400-200",
        "label": data_root / "testlabel-l0p500-a1p400-200",
        "conv_checkpoint": experiment_root / "models" / "convguided_20260818_l0p500_a1p400" / "convguided_filter_best_20260818_l0p500_a1p400.pth",
        "rf_checkpoint": experiment_root / "models" / "rf_3stage_20260818_l0p500_a1p400" / "rf_3stage_best_20260818_l0p500_a1p400.pth",
    }


def mean_metrics(metrics):
    if not metrics:
        raise ValueError("No test images were evaluated.")
    return {
        "mse": sum(item["mse"] for item in metrics) / len(metrics),
        "psnr": sum(item["psnr"] for item in metrics) / len(metrics),
    }


def _crop_to_even(tensor):
    return tensor[:, :, : tensor.shape[-2] - tensor.shape[-2] % 2,
                  : tensor.shape[-1] - tensor.shape[-1] % 2]


def _load_state(model, checkpoint_path, torch):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def _image_files(directory):
    return sorted(path for path in Path(directory).iterdir() if path.is_file())


def _validate_paths(paths):
    for name, path in paths.items():
        if name.endswith("checkpoint"):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {name}: {path}")
        elif not path.is_dir():
            raise FileNotFoundError(f"Missing test directory: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=r"D:\nuoqi\S2R-HDR\data\20260818")
    parser.add_argument("--experiment-root", default=r"D:\nuoqi\halo-free\Decoupled_Learning\20260815_edgepreserving")
    parser.add_argument("--rf-checkpoint", default=None, help="Override the refine checkpoint used for evaluation.")
    parser.add_argument("--conv-checkpoint", default=None, help="Override the ConvGuided checkpoint used for evaluation.")
    parser.add_argument("--rf-source", default=None, help="Python file defining the refine model architecture.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm
    import train_convguided_20260818_l0p500_a1p400 as conv_source
    if args.rf_source:
        source_path = Path(args.rf_source)
        spec = importlib.util.spec_from_file_location("rf_eval_source", source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load refine source: {source_path}")
        rf_source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rf_source)
    else:
        import train_rf_3stagev3_20260818_l0p500_a1p400 as rf_source

    paths = build_test_paths(args.data_root, args.experiment_root)
    if args.rf_checkpoint:
        paths["rf_checkpoint"] = Path(args.rf_checkpoint)
    if args.conv_checkpoint:
        paths["conv_checkpoint"] = Path(args.conv_checkpoint)
    _validate_paths(paths)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.experiment_root) / "evaluation_20260819"
    output_dir.mkdir(parents=True, exist_ok=True)

    lr_files = _image_files(paths["lr"])
    origin_files = _image_files(paths["origin"])
    label_files = _image_files(paths["label"])
    if not lr_files or not (len(lr_files) == len(origin_files) == len(label_files)):
        raise ValueError(f"Mismatched test files: LR={len(lr_files)}, origin={len(origin_files)}, label={len(label_files)}")

    device = torch.device("cuda:0")
    conv_model = conv_source.DeepGuidedFilter(radius=1, eps=1e-8).to(device)
    rf_model = rf_source.CascadeRefine3Stage(base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True).to(device)
    _load_state(conv_model, paths["conv_checkpoint"], torch)
    _load_state(rf_model, paths["rf_checkpoint"], torch)
    conv_model.eval()
    rf_model.eval()

    conv_metrics = []
    rf_metrics = []
    with torch.no_grad():
        for index, (lr_path, origin_path, label_path) in enumerate(tqdm(zip(lr_files, origin_files, label_files), total=len(lr_files), desc="Evaluating")):
            lr = cv2.imread(str(lr_path), cv2.IMREAD_UNCHANGED).astype("float32")
            origin = cv2.imread(str(origin_path), cv2.IMREAD_UNCHANGED).astype("float32")
            label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED).astype("float32")
            lr = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)
            origin = _crop_to_even(torch.from_numpy(origin).unsqueeze(0).unsqueeze(0)).to(device)
            label = _crop_to_even(torch.from_numpy(label).unsqueeze(0).unsqueeze(0)).to(device)

            conv_output = conv_model(lr - 0.5, origin - 0.5)
            conv_mse = torch.mean((conv_output - label) ** 2).item()
            conv_psnr = 100.0 if conv_mse == 0 else 10 * math.log10(1 / conv_mse)
            conv_metrics.append({"mse": conv_mse, "psnr": conv_psnr})

            rf_output = rf_model(origin - 0.5, conv_output - 0.5)["final"] + 0.5
            rf_output = torch.clamp(rf_output, 0, 1)
            rf_mse = torch.mean((rf_output - label) ** 2).item()
            rf_psnr = 100.0 if rf_mse == 0 else 10 * math.log10(1 / rf_mse)
            rf_metrics.append({"mse": rf_mse, "psnr": rf_psnr})

            preview = (rf_output[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / f"{index:04d}_{origin_path.stem}_rf.png"), preview)

    conv_summary = mean_metrics(conv_metrics)
    rf_summary = mean_metrics(rf_metrics)
    report = output_dir / "metrics.txt"
    report.write_text(
        "ConvGuided\n"
        f"MSE: {conv_summary['mse']:.8f}\nPSNR: {conv_summary['psnr']:.4f} dB\n\n"
        "RF final\n"
        f"MSE: {rf_summary['mse']:.8f}\nPSNR: {rf_summary['psnr']:.4f} dB\n",
        encoding="utf-8",
    )
    print(report.read_text(encoding="utf-8"))
    print(f"RF previews: {output_dir}")


if __name__ == "__main__":
    main()
