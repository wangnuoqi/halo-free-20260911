from __future__ import annotations

import argparse
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):  # type: ignore
        return iterable


VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_DATA_ROOT = Path(r"D:\nuoqi\halo-free\data_bsds\BSD68")
DEFAULT_EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = DEFAULT_EXPERIMENT_ROOT / "models" / "refine_gray" / "sigma25" / "refine_gray_sigma25_best.pth"


class Stage1DeltaNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, kernel_size=1, bias=False),
        )

    def forward(self, x_raw: torch.Tensor, x_coarse: torch.Tensor) -> torch.Tensor:
        g, _, _ = compute_sobel_grad(x_coarse, normalize=True)
        return self.net(torch.cat([x_raw, x_coarse, g], dim=1))


class ABAdjustNet(nn.Module):
    def __init__(self, base_channels: int = 32, a_scale: float = 0.2, b_scale: float = 0.2, use_delta_as_input: bool = True):
        super().__init__()
        self.a_scale = a_scale
        self.b_scale = b_scale
        self.use_delta_as_input = use_delta_as_input
        in_channels = 1 + 1 + 1 + (1 if use_delta_as_input else 0)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 2, kernel_size=1, bias=True),
        )

    def forward(self, x_raw: torch.Tensor, x_coarse_k: torch.Tensor, delta_prev: torch.Tensor | None = None):
        g, _, _ = compute_sobel_grad(x_coarse_k, normalize=True)
        if self.use_delta_as_input:
            if delta_prev is None:
                raise RuntimeError("delta_prev is required when use_delta_as_input=True.")
            features = torch.cat([x_raw, x_coarse_k, g, delta_prev], dim=1)
        else:
            features = torch.cat([x_raw, x_coarse_k, g], dim=1)
        a_raw, b_raw = torch.chunk(self.net(features), 2, dim=1)
        a = 1.0 + self.a_scale * torch.tanh(a_raw)
        b = self.b_scale * torch.tanh(b_raw)
        return a, b


class CascadeRefine3Stage(nn.Module):
    def __init__(self, base_channels: int = 32, a_scale: float = 0.2, b_scale: float = 0.2, use_delta_as_input: bool = True):
        super().__init__()
        self.stage1 = Stage1DeltaNet(base_channels=base_channels)
        self.stage2 = ABAdjustNet(base_channels=base_channels, a_scale=a_scale, b_scale=b_scale, use_delta_as_input=use_delta_as_input)
        self.stage3 = ABAdjustNet(base_channels=base_channels, a_scale=a_scale, b_scale=b_scale, use_delta_as_input=use_delta_as_input)

    def forward(self, x_raw: torch.Tensor, x_coarse0: torch.Tensor) -> Dict[str, torch.Tensor]:
        delta1 = self.stage1(x_raw, x_coarse0)
        x_coarse1 = x_coarse0 + delta1
        a2, b2 = self.stage2(x_raw, x_coarse1, delta1)
        delta2 = a2 * delta1 + b2
        x_coarse2 = x_coarse0 + delta2
        a3, b3 = self.stage3(x_raw, x_coarse2, delta2)
        delta3 = a3 * delta2 + b3
        return {
            "delta1": delta1,
            "coarse1": x_coarse1,
            "delta2": delta2,
            "coarse2": x_coarse2,
            "delta3": delta3,
            "final": x_coarse0 + delta3,
            "a2": a2,
            "b2": b2,
            "a3": a3,
            "b3": b3,
        }


def to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    if x.shape[1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def compute_sobel_grad(x: torch.Tensor, eps: float = 1e-6, normalize: bool = False):
    gray = to_gray(x)
    sobel_x = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]], device=x.device, dtype=x.dtype)
    sobel_y = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]], device=x.device, dtype=x.dtype)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + eps)
    if normalize:
        g = g / (g.amax(dim=(2, 3), keepdim=True) + eps)
    return g, gx, gy


def batched_otsu_threshold(g: torch.Tensor, num_bins: int = 256, eps: float = 1e-6, min_delta: float = 1e-6) -> torch.Tensor:
    if g.dim() != 4 or g.shape[1] != 1:
        raise ValueError("g must be (B,1,H,W)")
    batch_size = g.shape[0]
    flat = g.flatten(1)
    g_min = flat.min(dim=1).values
    g_max = flat.max(dim=1).values
    g_span = g_max - g_min
    span = g_span.clamp_min(eps)
    norm = ((flat - g_min[:, None]) / span[:, None]).clamp(0.0, 1.0)
    bin_idx = (norm * (num_bins - 1)).long()
    offsets = (torch.arange(batch_size, device=g.device) * num_bins)[:, None]
    hist = torch.bincount((bin_idx + offsets).reshape(-1), minlength=batch_size * num_bins)
    hist = hist.reshape(batch_size, num_bins).to(g.dtype)
    prob = hist / hist.sum(dim=1, keepdim=True).clamp_min(eps)
    centers = torch.linspace(0.0, 1.0, steps=num_bins, device=g.device, dtype=g.dtype).view(1, -1)
    omega = torch.cumsum(prob, dim=1)
    mu = torch.cumsum(prob * centers, dim=1)
    mu_total = mu[:, -1:]
    denom = (omega * (1.0 - omega)).clamp_min(eps)
    sigma_between = ((mu_total * omega - mu) ** 2) / denom
    sigma_between[:, 0] = 0.0
    sigma_between[:, -1] = 0.0
    best_bin = torch.argmax(sigma_between, dim=1)
    thresh = g_min + centers[0, best_bin] * span
    valid_span = g_span > min_delta
    return torch.where(valid_span, torch.minimum(thresh, g_max - min_delta), g_max)


def per_channel_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str = "l1") -> torch.Tensor:
    if loss_type == "l1":
        return (pred - target).abs()
    if loss_type == "l2":
        return (pred - target).pow(2)
    if loss_type == "huber":
        return F.smooth_l1_loss(pred, target, reduction="none")
    raise ValueError("loss_type must be 'l1' | 'l2' | 'huber'")


def masked_reduce(per: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    if mask is None:
        return per.mean()
    per = per * mask
    return per.sum() / mask.sum().clamp_min(eps)


def edge_guided_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    thresh_mode: str = "otsu",
    thresh: float = 0.9,
    dilate: int = 0,
    loss_type: str = "l1",
    eps: float = 1e-6,
    normalize_grad: bool = True,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shape mismatch: {pred.shape} vs {target.shape}")
    g, _, _ = compute_sobel_grad(target, eps=eps, normalize=normalize_grad)
    if thresh_mode == "fixed":
        edge_mask = (g > torch.as_tensor(thresh, device=g.device, dtype=g.dtype)).float()
    elif thresh_mode == "percentile":
        q = torch.quantile(g.flatten(1), torch.as_tensor(thresh, device=g.device), dim=1)
        edge_mask = (g > q.view(-1, 1, 1, 1)).float()
    elif thresh_mode == "otsu":
        t = batched_otsu_threshold(g, num_bins=256, eps=eps, min_delta=eps)
        edge_mask = (g > t.view(-1, 1, 1, 1)).float()
        edge_count = edge_mask.flatten(1).sum(dim=1)
        no_edge = edge_count <= 0
        if no_edge.any():
            flat_g = g.flatten(1)
            fallback = torch.zeros_like(flat_g)
            fallback.scatter_(1, flat_g.argmax(dim=1, keepdim=True), 1.0)
            flat_mask = edge_mask.flatten(1)
            flat_mask[no_edge] = fallback[no_edge]
            edge_mask = flat_mask.view_as(edge_mask)
    else:
        raise ValueError("thresh_mode must be 'fixed' | 'percentile' | 'otsu'")
    if dilate and dilate > 0:
        edge_mask = F.max_pool2d(edge_mask, kernel_size=dilate * 2 + 1, stride=1, padding=dilate)
    edge_mask = edge_mask.detach()
    smooth_mask = (1.0 - edge_mask).clamp(0.0, 1.0)
    per_pix = per_channel_loss(pred, target, loss_type=loss_type)
    if per_pix.shape[1] > 1:
        per_pix = per_pix.mean(dim=1, keepdim=True)
    edge_pix = masked_reduce(per_pix, edge_mask, eps=eps)
    smooth_pix = masked_reduce(per_pix, smooth_mask, eps=eps)
    _, gpx, gpy = compute_sobel_grad(pred, eps=eps, normalize=normalize_grad)
    _, gtx, gty = compute_sobel_grad(target, eps=eps, normalize=normalize_grad)
    edge_grad = masked_reduce((gpx - gtx).abs() + (gpy - gty).abs(), edge_mask, eps=eps)
    ratio_edge = (edge_grad / (edge_pix + eps)).detach().clamp_min(eps)
    ratio_smooth = (smooth_pix / (edge_pix + eps)).detach().clamp_min(eps)
    return edge_pix + edge_grad / ratio_edge + smooth_pix / ratio_smooth


def cascade_loss(outputs: Dict[str, torch.Tensor], x_coarse0: torch.Tensor, gt: torch.Tensor, **kwargs):
    delta_gt = gt - x_coarse0
    l1 = edge_guided_loss(outputs["delta1"], delta_gt, **kwargs)
    l2 = edge_guided_loss(outputs["delta2"], delta_gt, **kwargs)
    l3 = edge_guided_loss(outputs["delta3"], delta_gt, **kwargs)
    return 0.33 * l1 + 0.33 * l2 + 0.33 * l3, (l1, l2, l3)


def normalize_path_for_runtime(path: str | os.PathLike | None) -> str | None:
    if path is None:
        return None
    path = str(path)
    if os.name == "nt":
        return path
    windows_drive_match = re.match(r"^([A-Za-z]):\\(.*)$", path)
    if not windows_drive_match:
        return path
    drive = windows_drive_match.group(1).lower()
    suffix = windows_drive_match.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{suffix}"


def read_gray01(path: str | os.PathLike) -> np.ndarray:
    image = imageio.imread(normalize_path_for_runtime(path))
    if image.ndim == 3:
        image = image[..., :3].astype(np.float32)
        image = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image


def rotate_landscape(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(image, k=-1))


def restore_orientation(image: np.ndarray, was_rotated: bool) -> np.ndarray:
    if was_rotated:
        image = np.rot90(image, k=1)
    return np.ascontiguousarray(image)


def crop_to_even(tensor: torch.Tensor) -> torch.Tensor:
    _, _, height, width = tensor.shape
    return tensor[:, :, : height - (height % 2), : width - (width % 2)]


def output_dir_for_sigma(experiment_root: str | os.PathLike, sigma: float) -> Path:
    return Path(experiment_root) / "eval_results-img-rf" / f"sigma{float(sigma):g}"


class GuidedGrayRefineEvalDataset(Dataset):
    """Full-image gray RF inputs: noisy raw, saved ConvGuided base, and clean GT."""

    def __init__(self, origin_input_dir: str | os.PathLike, input_dir: str | os.PathLike,
                 label_dir: str | os.PathLike, sigma: float):
        self.origin_input_dir = Path(normalize_path_for_runtime(origin_input_dir))
        self.input_dir = Path(normalize_path_for_runtime(input_dir))
        self.label_dir = Path(normalize_path_for_runtime(label_dir))
        self.sigma = float(sigma)
        names = self._names(self.origin_input_dir) & self._names(self.input_dir) & self._names(self.label_dir)
        self.names = sorted(names)
        if not self.names:
            raise RuntimeError(f"No matching images found in {self.origin_input_dir}, {self.input_dir}, {self.label_dir}")

    @staticmethod
    def _names(folder: Path) -> set[str]:
        return {path.name for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTS}

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        name = self.names[idx]
        clean = read_gray01(self.origin_input_dir / name)
        label = read_gray01(self.label_dir / name)
        coarse_input = read_gray01(self.input_dir / name)
        rng = np.random.RandomState(0)
        origin_input = clean + rng.normal(0.0, self.sigma / 255.0, clean.shape).astype(np.float32)
        if clean.shape[0] > clean.shape[1]:
            origin_input = rotate_landscape(origin_input)
            label = rotate_landscape(label)
        height = min(origin_input.shape[0], coarse_input.shape[0], label.shape[0])
        width = min(origin_input.shape[1], coarse_input.shape[1], label.shape[1])
        return {
            "origin_input_hr": torch.from_numpy(np.ascontiguousarray(origin_input[:height, :width][None])),
            "input_hr": torch.from_numpy(np.ascontiguousarray(coarse_input[:height, :width][None])),
            "target_img": torch.from_numpy(np.ascontiguousarray(label[:height, :width][None])),
            "filename": name,
            "was_rotated": False,
            "input_height_hr": height,
            "input_width_hr": width,
        }


def var_custom_collate_with_names(batch: List[Dict[str, object]]):
    min_h_hr = min(int(item["input_height_hr"]) for item in batch)
    min_w_hr = min(int(item["input_width_hr"]) for item in batch)
    raw = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    coarse = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    label = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    names = []
    was_rotated = []
    for idx, item in enumerate(batch):
        raw[idx] = item["origin_input_hr"][:, :min_h_hr, :min_w_hr]
        coarse[idx] = item["input_hr"][:, :min_h_hr, :min_w_hr]
        label[idx] = item["target_img"][:, :min_h_hr, :min_w_hr]
        names.append(str(item["filename"]))
        was_rotated.append(bool(item.get("was_rotated", False)))
    return raw, coarse, label, names, was_rotated


def save_gray_prediction(prediction: torch.Tensor, output_path: str | os.PathLike, was_rotated: bool = False, overwrite: bool = True) -> bool:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return False
    image = prediction.detach().cpu().squeeze(0).numpy()
    image = restore_orientation(image, was_rotated)
    image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, image)
    return True


def parse_gpu_ids(gpu_ids: str) -> List[int]:
    if not gpu_ids:
        return []
    return [int(item.strip()) for item in gpu_ids.split(",") if item.strip()]


def resolve_device(device_name: str, gpu_ids: List[int]) -> torch.device:
    if str(device_name).startswith("cuda") and torch.cuda.is_available():
        if ":" in str(device_name):
            return torch.device(device_name)
        return torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cuda")
    return torch.device("cpu")


def load_state_dict_flexible(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    else:
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)


def load_checkpoint_flexible(model: torch.nn.Module, checkpoint_path: str | os.PathLike, device: torch.device) -> None:
    checkpoint = torch.load(normalize_path_for_runtime(checkpoint_path), map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    load_state_dict_flexible(model, state_dict)


def calc_psnr(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    mse = torch.mean((pred - target) ** 2).item()
    return (100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)), mse


def gaussian_kernel(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def calc_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shape mismatch: {pred.shape} vs {target.shape}")
    channels = pred.shape[1]
    kernel = gaussian_kernel(window_size, sigma, channels, pred.device, pred.dtype)
    padding = window_size // 2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_pred = F.conv2d(pred, kernel, padding=padding, groups=channels)
    mu_target = F.conv2d(target, kernel, padding=padding, groups=channels)
    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, kernel, padding=padding, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, kernel, padding=padding, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, kernel, padding=padding, groups=channels) - mu_pred_target

    ssim_map = ((2.0 * mu_pred_target + c1) * (2.0 * sigma_pred_target + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    )
    return ssim_map.mean().item()


def build_loader(args: argparse.Namespace) -> DataLoader:
    dataset = GuidedGrayRefineEvalDataset(
        origin_input_dir=args.origin_input_test,
        input_dir=args.input_folder_test,
        label_dir=args.input_folder_test_label,
        sigma=args.sigma,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=var_custom_collate_with_names,
        pin_memory=str(args.device).startswith("cuda"),
        drop_last=False,
    )


def maybe_progress(iterable: Iterable, *, desc: str, disable: bool = False):
    if disable:
        return iterable
    return tqdm(iterable, desc=desc, dynamic_ncols=True)


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    args.origin_input_test = normalize_path_for_runtime(args.origin_input_test)
    args.input_folder_test = normalize_path_for_runtime(args.input_folder_test)
    args.input_folder_test_label = normalize_path_for_runtime(args.input_folder_test_label)
    args.checkpoint = normalize_path_for_runtime(args.checkpoint)
    args.output_dir = normalize_path_for_runtime(args.output_dir)

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    device = resolve_device(args.device, gpu_ids)
    loader = build_loader(args)
    model = CascadeRefine3Stage(
        base_channels=args.base_channels,
        a_scale=args.a_scale,
        b_scale=args.b_scale,
        use_delta_as_input=not args.no_delta_as_input,
    )
    if device.type == "cuda":
        model = nn.DataParallel(model, device_ids=gpu_ids or None).to(device)
    else:
        model = model.to(device)
    load_checkpoint_flexible(model, args.checkpoint, device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_loss = 0.0
    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for raw, coarse, label, names, was_rotated in maybe_progress(loader, desc="Testing", disable=args.no_progress):
            raw = crop_to_even(raw).to(device)
            coarse = crop_to_even(coarse).to(device)
            label = crop_to_even(label).to(device)
            raw = raw - 0.5
            coarse = coarse - 0.5
            label_centered = label - 0.5
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            outputs = model(raw, coarse)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.time() - start_time
            loss, _ = cascade_loss(
                outputs,
                coarse,
                label_centered,
                thresh_mode=args.thresh_mode,
                thresh=args.thresh,
                dilate=args.dilate,
                loss_type=args.loss_type,
                normalize_grad=True,
            )
            output = torch.clamp(outputs["final"] + 0.5, 0.0, 1.0)
            label_eval = torch.clamp(label_centered + 0.5, 0.0, 1.0)
            for idx, name in enumerate(names):
                save_gray_prediction(output[idx], output_dir / name, was_rotated=was_rotated[idx], overwrite=args.overwrite)
            psnr, mse = calc_psnr(output, label_eval)
            ssim = calc_ssim(output, label_eval)
            total_mse += mse
            total_psnr += psnr
            total_ssim += ssim
            total_loss += loss.item()
            total_images += raw.shape[0]

    num_batches = max(1, len(loader))
    metrics = {
        "avg_mse": total_mse / num_batches,
        "avg_psnr": total_psnr / num_batches,
        "avg_ssim": total_ssim / num_batches,
        "avg_loss": total_loss / num_batches,
        "avg_infer_time_ms": total_time / max(1, total_images) * 1000.0,
        "images": float(total_images),
    }
    print(
        "Test Finished - "
        f"Average PSNR: {metrics['avg_psnr']:.2f} dB, "
        f"Average SSIM: {metrics['avg_ssim']:.4f}, "
        f"Average MSE: {metrics['avg_mse']:.6f}, "
        f"Average Loss: {metrics['avg_loss']:.6f}"
    )
    print(f"Average inference time per image: {metrics['avg_infer_time_ms']:.2f} ms")
    print(f"Saved outputs to: {output_dir}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the grayscale CascadeRefine3Stage on full-resolution BSD68 images.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--origin-input-test", default=None)
    parser.add_argument("--input-folder-test", default=None)
    parser.add_argument("--input-folder-test-label", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--a-scale", type=float, default=0.2)
    parser.add_argument("--b-scale", type=float, default=0.2)
    parser.add_argument("--no-delta-as-input", action="store_true")
    parser.add_argument("--thresh-mode", default="otsu", choices=["fixed", "percentile", "otsu"])
    parser.add_argument("--thresh", type=float, default=0.6)
    parser.add_argument("--dilate", type=int, default=0)
    parser.add_argument("--loss-type", default="l1", choices=["l1", "l2", "huber"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    data_root = Path(normalize_path_for_runtime(args.data_root))
    if args.origin_input_test is None:
        args.origin_input_test = str(data_root)
    if args.input_folder_test is None:
        args.input_folder_test = str(
            Path(normalize_path_for_runtime(DEFAULT_EXPERIMENT_ROOT)) / "eval_results-img" / f"sigma{args.sigma:g}"
        )
    if args.input_folder_test_label is None:
        args.input_folder_test_label = str(data_root)
    if args.checkpoint is None:
        args.checkpoint = str(
            Path(normalize_path_for_runtime(DEFAULT_EXPERIMENT_ROOT))
            / "models" / "refine_gray" / f"sigma{args.sigma:g}" / f"refine_gray_sigma{args.sigma:g}_best.pth"
        )
    if args.output_dir is None:
        args.output_dir = str(output_dir_for_sigma(
            normalize_path_for_runtime(DEFAULT_EXPERIMENT_ROOT), args.sigma
        ))
    return args


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
