"""Evaluate all paired OpenClipart Q30 models with one fixed RGB protocol.

Every method is evaluated on the same 466 images from ``test_list.txt``.
Outputs and targets are treated as 512x512 RGB images in [0, 1].  The
single-channel exp3-a model predicts Y and reuses the input JPEG chroma.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except ImportError:
    skimage_ssim = None


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "openclipart_5000" / "jpeg_pairs_512"
DEFAULT_LIST = HERE / "openclipart_5000"
DEFAULT_OUTPUT = HERE / "evaluation" / "q30_all_models"
DEFAULT_JOINT = HERE / "train_clipart_joint" / "ckpt"
DEFAULT_BASELINES = HERE / "baselines"
BASELINE_CHECKPOINTS = {}


def import_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_state(model, path: Path, device, key_candidates=("model_state_dict", "model"), saved=None):
    if saved is None:
        saved = torch.load(path, map_location=device, weights_only=False)
    state = saved
    if isinstance(saved, dict):
        for key in key_candidates:
            if key in saved and isinstance(saved[key], dict):
                state = saved[key]
                break
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return saved


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def read_ycbcr(path: Path):
    with Image.open(path) as image:
        ycbcr = np.asarray(image.convert("YCbCr"), dtype=np.float32) / 255.0
    return ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]


def to_tensor(image: np.ndarray, device):
    return torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)


def to_batch(images, device):
    return torch.from_numpy(np.stack([image.transpose(2, 0, 1) for image in images])).to(device)


def resize_half_y(y: np.ndarray) -> np.ndarray:
    height, width = y.shape
    resized = Image.fromarray(np.clip(y * 255.0, 0, 255).astype(np.uint8), mode="L")
    return np.asarray(resized.resize((max(2, width // 2), max(2, height // 2)), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0


def resize_half_rgb(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    image = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))
    size = (max(1, width // 2), max(1, height // 2))
    return np.asarray(image.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0


def y_to_rgb(y, cb, cr):
    ycbcr = np.stack([y, cb, cr], axis=-1)
    # PIL's YCbCr conversion is the same convention used for input decoding.
    image = Image.fromarray(np.clip(ycbcr * 255.0, 0, 255).round().astype(np.uint8), mode="YCbCr")
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def ssim_gray_or_rgb(pred: np.ndarray, target: np.ndarray) -> float:
    """Standard 11x11-window SSIM averaged over RGB channels."""
    if skimage_ssim is None:
        raise RuntimeError("SSIM requires scikit-image: pip install scikit-image")
    return float(skimage_ssim(target, pred, channel_axis=2, data_range=1.0,
                              gaussian_weights=True, sigma=1.5,
                              use_sample_covariance=False))


def psnrb(pred: np.ndarray, target: np.ndarray) -> float:
    """RGB PSNR-B, including the JPEG 8x8 blocking-effect penalty."""
    total = 0.0
    h, w = pred.shape[:2]
    for channel in range(3):
        x, y = pred[..., channel], target[..., channel]
        mse = float(np.mean((x - y) ** 2))
        horizontal_boundary = np.mean((x[:, 7:w - 1:8] - x[:, 8:w:8]) ** 2)
        vertical_boundary = np.mean((x[7:h - 1:8, :] - x[8:h:8, :]) ** 2)
        horizontal_nonboundary = np.delete(np.arange(w - 1), np.arange(7, w - 1, 8))
        vertical_nonboundary = np.delete(np.arange(h - 1), np.arange(7, h - 1, 8))
        boundary = (horizontal_boundary * h * max(w // 8 - 1, 0) +
                    vertical_boundary * w * max(h // 8 - 1, 0))
        nonboundary = 0.0
        count = 0
        if horizontal_nonboundary.size:
            nonboundary += float(np.mean((x[:, horizontal_nonboundary] - x[:, horizontal_nonboundary + 1]) ** 2))
            count += 1
        if vertical_nonboundary.size:
            nonboundary += float(np.mean((x[vertical_nonboundary, :] - x[vertical_nonboundary + 1, :]) ** 2))
            count += 1
        nonboundary = nonboundary / max(count, 1)
        n_boundary = h * max(w // 8 - 1, 0) + w * max(h // 8 - 1, 0)
        bef = (math.log2(8) / math.log2(min(h, w))) * max(boundary / max(n_boundary, 1) - nonboundary, 0.0)
        total += 10.0 * math.log10(1.0 / max(mse + bef, 1e-12))
    return total / 3.0


def psnr(mse: float) -> float:
    return 100.0 if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def metric_row(pred: np.ndarray, target: np.ndarray):
    pred = np.clip(pred, 0.0, 1.0).astype(np.float32)
    target = np.clip(target, 0.0, 1.0).astype(np.float32)
    error = pred - target
    mse = float(np.mean(error * error))
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(error))),
        "psnr": psnr(mse),
        "ssim": ssim_gray_or_rgb(pred, target),
        "psnr_b": psnrb(pred, target),
    }


def save_rgb(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255.0, 0, 255).round().astype(np.uint8), mode="RGB").save(path)


def load_ours(exp: str, root: Path, device, multi_gpu=False, checkpoint_override=None):
    safe_name = exp.replace("-", "_")
    common = import_from(root / "train_clipart_joint" / "clipart_conv_base_common.py", f"common_{safe_name}")
    refine_defs = import_from(root / "train_clipart_joint" / "clipart_refine_models.py", f"refine_{safe_name}")
    channels = 1 if exp == "exp3-a" else 3
    base_path = Path(checkpoint_override) if checkpoint_override is not None else root / "train_clipart_joint" / "ckpt" / f"{exp}_joint" / "joint_best.pth"
    model = {
        "base": common.DeepGuidedFilter(channels).to(device),
        "refine": refine_defs.CascadeRefine3Stage(channels=channels).to(device),
        "channels": channels,
        "checkpoint": base_path,
    }
    saved = torch.load(base_path, map_location=device, weights_only=False)
    model["base"].load_state_dict({k.removeprefix("module."): v for k, v in saved["base_state_dict"].items()}, strict=True)
    model["refine"].load_state_dict({k.removeprefix("module."): v for k, v in saved["refine_state_dict"].items()}, strict=True)
    if multi_gpu:
        model["base"] = torch.nn.DataParallel(model["base"])
        model["refine"] = torch.nn.DataParallel(model["refine"])
    model["base"].eval(); model["refine"].eval()
    return model


def predict_ours(model, input_paths, device):
    if model["channels"] == 1:
        ycbcr = [read_ycbcr(path) for path in input_paths]
        ys = [item[0] for item in ycbcr]
        hr = torch.from_numpy(np.stack(ys)[:, None]).to(device)
        lr = torch.from_numpy(np.stack([resize_half_y(y) for y in ys])[:, None]).to(device)
        with torch.no_grad():
            coarse = model["base"](lr - 0.5, hr - 0.5)
            final = model["refine"](hr - 0.5, coarse - 0.5)["final"]
        output_y = (final + 0.5).clamp(0, 1)[:, 0].cpu().numpy()
        return [y_to_rgb(output_y[i], ycbcr[i][1], ycbcr[i][2]) for i in range(len(input_paths))]
    rgb = [read_rgb(path) for path in input_paths]
    hr = to_batch(rgb, device)
    lr = to_batch([resize_half_rgb(image) for image in rgb], device)
    with torch.no_grad():
        coarse = model["base"](lr - 0.5, hr - 0.5)
        final = model["refine"](hr - 0.5, coarse - 0.5)["final"]
    return [(final + 0.5).clamp(0, 1)[i].permute(1, 2, 0).cpu().numpy() for i in range(len(input_paths))]


def load_baselines(root: Path, device, methods, multi_gpu=False):
    models = {}
    checkpoint_map = {}
    BASELINE_CHECKPOINTS.clear()
    if "ARCNN" in methods:
        module = import_from(root / "baselines/ARCNN-pytorch/model.py", "arcnn_model")
        model = module.ARCNN().to(device)
        checkpoint = root / "baselines/ARCNN-pytorch/checkpoints/q30_protocol160/best.pth"
        load_state(model, checkpoint, device)
        models["ARCNN"] = model
        checkpoint_map["ARCNN"] = str(checkpoint)
    if "FBCNN" in methods:
        # FBCNN imports torchvision internally; keep its project import isolated.
        module = import_from(root / "baselines/FBCNN/models/network_fbcnn.py", "fbcnn_model")
        model = module.FBCNN(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4,
                             act_mode="BR", downsample_mode="strideconv",
                             upsample_mode="convtranspose").to(device)
        checkpoint = root / "baselines/ckpt/fbcnn_rgb_q30_protocol160/fbcnn_best.pth"
        load_state(model, checkpoint, device,
                   key_candidates=("model_state_dict", "model"))
        models["FBCNN"] = model
        checkpoint_map["FBCNN"] = str(checkpoint)
    if "SwinIR" in methods:
        module = import_from(root / "baselines/SwinIR_downloaded/SwinIR-main/models/network_swinir.py", "swin_model")
        checkpoint = root / "baselines/SwinIR_downloaded/SwinIR-main/checkpoints/q30_protocol160/best.pth"
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        patch_size = int(saved.get("config", {}).get("patch_size", 126))
        model_size = ((patch_size + 6) // 7) * 7
        model = module.SwinIR(upscale=1, in_chans=3, img_size=model_size, window_size=7, img_range=255., depths=[6] * 6, embed_dim=180, num_heads=[6] * 6, mlp_ratio=2, upsampler='', resi_connection='1conv').to(device)
        load_state(model, checkpoint, device, saved=saved)
        models["SwinIR"] = model
        checkpoint_map["SwinIR"] = str(checkpoint)
    if "VIVNet" in methods:
        module = import_from(
            root / "baselines/VIVNet/basicsr/models/archs/VIVNet_arch.py",
            "vivnet_arch_for_evaluation",
        )
        model = module.VIVNet(
            inp_channels=3,
            out_channels=3,
            dim=32,
            num_blocks=[1, 1, 1],
            num_refinement_blocks=4,
            ffn_expansion_factor=3,
            bias=False,
        ).to(device)
        checkpoint = root / (
            "baselines/VIVNet/experiments/"
            "OpenClipart-Q30-VIVNet-T-protocol160-3gpu/models/net_g_latest.pth"
        )
        load_state(model, checkpoint, device, key_candidates=("params",))
        models["VIVNet"] = model
        checkpoint_map["VIVNet"] = str(checkpoint)
    BASELINE_CHECKPOINTS.update(checkpoint_map)
    if multi_gpu:
        models = {name: torch.nn.DataParallel(model) for name, model in models.items()}
    for model in models.values():
        model.eval()
    return models


def predict_baseline(name, model, input_paths, device):
    rgb = [read_rgb(path) for path in input_paths]
    tensor = to_batch(rgb, device)
    with torch.no_grad():
        if name == "FBCNN":
            quality = tensor.new_full((tensor.shape[0], 1), 0.7)
            output = model(tensor, quality)[0]
        elif name == "SwinIR":
            output = model(tensor)
        elif name == "VIVNet":
            _, _, height, width = tensor.shape
            pad_height = (32 - height % 32) % 32
            pad_width = (32 - width % 32) % 32
            padded = F.pad(tensor, (0, pad_width, 0, pad_height), mode="reflect")
            output = model(padded)[:, :, :height, :width]
        else:
            output = model(tensor)
    output = output.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    return [output[i] for i in range(len(input_paths))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=HERE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods", nargs="+", default=["ARCNN", "FBCNN", "SwinIR", "exp3-a", "exp3-b"])
    parser.add_argument("--exp3-b-checkpoint", type=Path, default=None,
                        help="Override the exp3-b joint checkpoint, for example the patch128 batch64 run.")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Evaluate only the first N test pairs (useful for smoke tests).")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of 512x512 images inferred together on the selected device.")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use DataParallel across all visible CUDA devices.")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    multi_gpu = bool(args.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1)
    if args.multi_gpu and not multi_gpu:
        raise RuntimeError("--multi-gpu requires at least two visible CUDA devices")
    data_root = args.project_root / "openclipart_5000" / "jpeg_pairs_512"
    list_path = args.project_root / "openclipart_5000" / "test_list.txt"
    stems = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    pairs = [(data_root / "jpeg_q30" / f"{stem}.jpg", data_root / "clean_png" / f"{stem}.png") for stem in stems]
    if len(pairs) != 466 or any(not a.is_file() or not b.is_file() for a, b in pairs):
        raise RuntimeError("Expected 466 complete Q30 test pairs")
    if args.max_images is not None:
        if args.max_images < 1:
            raise ValueError("--max-images must be positive")
        pairs = pairs[:args.max_images]
    baseline_models = load_baselines(args.project_root, device, args.methods, multi_gpu) if any(m in args.methods for m in ("ARCNN", "FBCNN", "SwinIR", "VIVNet")) else {}
    ours = {m: load_ours(m, args.project_root, device, multi_gpu,
                         checkpoint_override=args.exp3_b_checkpoint if m == "exp3-b" else None)
            for m in args.methods if m in ("exp3-a", "exp3-b")}
    rows = []
    timing = {}
    for method in args.methods:
        method_dir = args.output_dir / method.replace("-", "_")
        method_wall_start = time.perf_counter()
        inference_seconds = 0.0
        processed_images = 0
        for batch_start in range(0, len(pairs), args.batch_size):
            batch = pairs[batch_start:batch_start + args.batch_size]
            input_paths = [item[0] for item in batch]
            inference_start = time.perf_counter()
            predictions = (predict_ours(ours[method], input_paths, device)
                           if method in ours else predict_baseline(method, baseline_models[method], input_paths, device))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - inference_start
            processed_images += len(batch)
            for offset, ((input_path, gt_path), pred) in enumerate(zip(batch, predictions)):
                index = batch_start + offset + 1
                gt = read_rgb(gt_path)
                metrics = metric_row(pred, gt)
                save_rgb(method_dir / f"{input_path.stem}.png", pred)
                rows.append({"method": method, "index": index, "image": input_path.stem, **metrics})
            index = batch_start + len(batch)
            if index == 1 or index % 50 == 0 or index == len(pairs):
                print(f"[{method}] {index}/{len(pairs)}", flush=True)
        wall_seconds = time.perf_counter() - method_wall_start
        print(f"[{method}] inference={inference_seconds:.3f}s ({inference_seconds / processed_images:.4f}s/image), "
              f"wall={wall_seconds:.3f}s ({wall_seconds / processed_images:.4f}s/image)", flush=True)
        timing[method] = {
            "inference_seconds": inference_seconds,
            "inference_seconds_per_image": inference_seconds / processed_images,
            "wall_seconds": wall_seconds,
            "wall_seconds_per_image": wall_seconds / processed_images,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for method in args.methods:
        subset = [row for row in rows if row["method"] == method]
        summary.append({"method": method, "images": len(subset), **{key: float(np.mean([row[key] for row in subset])) for key in ("mse", "mae", "psnr", "ssim", "psnr_b")}, **timing[method]})
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (args.output_dir / "evaluation_config.json").write_text(json.dumps({"data_root": str(data_root), "test_list": str(list_path), "count": len(pairs), "methods": args.methods, "checkpoints": {**BASELINE_CHECKPOINTS, **{m: str(ours[m]["checkpoint"]) for m in ours}}, "batch_size": args.batch_size, "multi_gpu": multi_gpu, "visible_cuda_devices": torch.cuda.device_count() if device.type == "cuda" else 0, "metrics": ["MSE", "MAE", "PSNR", "SSIM", "PSNR-B"], "timing": ["inference_seconds", "inference_seconds_per_image", "wall_seconds", "wall_seconds_per_image"], "rgb_output": True, "inference_geometry": "full image with architecture-specific padding where required"}, indent=2), encoding="utf-8")
    print(f"Saved: {args.output_dir / 'summary.csv'}")
    for row in summary: print(row)


if __name__ == "__main__":
    main()
