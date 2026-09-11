import argparse
import csv
import math
import os
import random
import re
import time

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

LAMBDA_MIN = 0.025
LAMBDA_MAX = 2.5
ALPHA_MIN = 0.8
ALPHA_MAX = 1.6
BASE_CHANNELS = 24

DEFAULT_X_LR_DIR = "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/testinput_downsampled"
DEFAULT_X_HR_DIR = "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/testinput"
DEFAULT_LABEL_DIR = "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/testlabel"
DEFAULT_MODEL_DIR = "/mnt/d/nuoqi/halo-free/Decoupled-params/model"
DEFAULT_OUTPUT_ROOT = "/mnt/d/nuoqi/halo-free/Decoupled-params/eval_outputs"
DEFAULT_SUMMARY_PATH = "/mnt/d/nuoqi/halo-free/Decoupled-params/eval_outputs/frozen_eval_summary.csv"

MODEL_FILENAMES = {
    "paraminput_freezeb": {
        "best": "convguided_filter_best_mix_paraminput_freezeb.pth",
        "last": "convguided_filter_last_mix_paraminput_freezeb.pth",
        "full": "convguided_filter_full_mix_paraminput_freezeb.pth",
    },
    "paramtransfer_freezefront": {
        "best": "convguided_filter_best_mix_paramtransfer_freezefront.pth",
        "last": "convguided_filter_last_mix_paramtransfer_freezefront.pth",
        "full": "convguided_filter_full_mix_paramtransfer_freezefront.pth",
    },
    "paramadapter": {
        "best": "convguided_filter_best_adapter_param.pth",
        "last": "convguided_filter_last_adapter_param.pth",
        "full": "convguided_filter_full_adapter_param.pth",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Single-GPU evaluation for frozen ConvGuidedFilter variants.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(MODEL_FILENAMES.keys()),
        choices=list(MODEL_FILENAMES.keys()),
        help="Frozen model variants to evaluate.",
    )
    parser.add_argument(
        "--model-kind",
        default="best",
        choices=["best", "last", "full"],
        help="Checkpoint kind to evaluate for each variant.",
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--x-lr-dir", default=DEFAULT_X_LR_DIR)
    parser.add_argument("--x-hr-dir", default=DEFAULT_X_HR_DIR)
    parser.add_argument("--label-dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0, help="Single CUDA device index.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-preds", action="store_true", help="Save predicted TIFF files.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args()


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))
        self.bn = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


def weights_init_identity(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        n_out, n_in, h, w = m.weight.data.size()
        if n_out < n_in:
            nn.init.xavier_uniform_(m.weight.data)
            return
        m.weight.data.zero_()
        ch, cw = h // 2, w // 2
        for i in range(n_in):
            m.weight.data[i, i, ch, cw] = 1.0
    elif classname.find("BatchNorm2d") != -1:
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)


class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1, norm=nn.BatchNorm2d):
        super().__init__()
        self.box_filter = nn.Conv2d(1, 1, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=1)
        self.conv_a = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
        )
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()
        n = self.box_filter(x_lr.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0))
        mean_x = self.box_filter(x_lr) / n
        mean_y = self.box_filter(y_lr) / n
        cov_xy = self.box_filter(x_lr * y_lr) / n - mean_x * mean_y
        var_x = self.box_filter(x_lr * x_lr) / n - mean_x * mean_x
        a = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        b = mean_y - a * mean_x
        mean_a = F.interpolate(a, (h_hrx, w_hrx), mode="bilinear", align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode="bilinear", align_corners=True)
        return mean_a * x_hr + mean_b


def build_paraminput_lr_net(norm=AdaptiveNorm, layer=5, in_channels=3):
    layers = [
        nn.Conv2d(in_channels, BASE_CHANNELS, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(BASE_CHANNELS),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    for l in range(1, layer):
        layers += [
            nn.Conv2d(BASE_CHANNELS, BASE_CHANNELS, kernel_size=3, stride=1, padding=2**l, dilation=2**l, bias=False),
            norm(BASE_CHANNELS),
            nn.LeakyReLU(0.2, inplace=True),
        ]
    layers += [
        nn.Conv2d(BASE_CHANNELS, BASE_CHANNELS, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(BASE_CHANNELS),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(BASE_CHANNELS, 1, kernel_size=1, stride=1, padding=0, dilation=1),
    ]
    net = nn.Sequential(*layers)
    net.apply(weights_init_identity)
    return net


class DeepGuidedFilterParamInput(nn.Module):
    def __init__(self, radius=1):
        super().__init__()
        self.lr = build_paraminput_lr_net()
        self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm)

    def forward(self, x_lr, x_hr, params):
        params = normalize_wls_params(params).to(x_lr.dtype) - 0.5
        param_maps = params[:, :, None, None].expand(-1, -1, x_lr.size(2), x_lr.size(3))
        x_lr_cond = torch.cat([x_lr, param_maps], dim=1)
        y_lr = self.lr(x_lr_cond)
        return self.gf(x_lr, y_lr, x_hr).clamp(0, 1)


def build_single_param_lr_layers(norm=AdaptiveNorm, layer=5):
    layers = [
        nn.Conv2d(1, BASE_CHANNELS, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(BASE_CHANNELS),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    for l in range(1, layer):
        layers += [
            nn.Conv2d(BASE_CHANNELS, BASE_CHANNELS, kernel_size=3, stride=1, padding=2**l, dilation=2**l, bias=False),
            norm(BASE_CHANNELS),
            nn.LeakyReLU(0.2, inplace=True),
        ]
    layers += [
        nn.Conv2d(BASE_CHANNELS, BASE_CHANNELS, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(BASE_CHANNELS),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(BASE_CHANNELS, 1, kernel_size=1, stride=1, padding=0, dilation=1),
    ]
    return layers


class LateParamModulator(nn.Module):
    def __init__(self, channels=BASE_CHANNELS, hidden=32):
        super().__init__()
        self.fc1_weight = nn.Linear(2, hidden)
        self.fc1_bias = nn.Linear(2, hidden)
        self.fc2_weight = nn.Linear(hidden, channels)
        self.fc2_bias = nn.Linear(hidden, channels)
        self.relu = nn.ReLU(inplace=True)
        self.inject_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        weights_init_identity(self.inject_conv)

    def forward(self, feat, params):
        weight = self.fc2_weight(self.relu(self.fc1_weight(params)))
        bias = self.fc2_bias(self.relu(self.fc1_bias(params)))
        feat = self.inject_conv(feat)
        outputs = []
        for n in range(feat.size(0)):
            feat_n = F.instance_norm(feat.narrow(0, n, 1), weight=weight[n], bias=bias[n])
            outputs.append(F.leaky_relu(feat_n, negative_slope=0.2, inplace=False))
        return torch.cat(outputs, dim=0)


class DeepGuidedFilterParamTransfer(nn.Module):
    def __init__(self, radius=1):
        super().__init__()
        lr_layers = build_single_param_lr_layers()
        self.lr_front = nn.Sequential(*lr_layers[:9])
        self.lr_back = nn.Sequential(*lr_layers[9:18])
        self.lr_out = lr_layers[18]
        self.param_mod = LateParamModulator(channels=BASE_CHANNELS, hidden=32)
        self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm)

        self.lr_front.apply(weights_init_identity)
        self.lr_back.apply(weights_init_identity)
        weights_init_identity(self.lr_out)

    def forward(self, x_lr, x_hr, params):
        params = normalize_wls_params(params).to(x_lr.dtype) - 0.5
        feat = self.lr_front(x_lr)
        feat = self.lr_back(feat)
        feat = self.param_mod(feat, params)
        y_lr = self.lr_out(feat)
        return self.gf(x_lr, y_lr, x_hr).clamp(0, 1)


def build_lr_backbone(norm=AdaptiveNorm, layer=5):
    layers = [
        nn.Conv2d(1, BASE_CHANNELS, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(BASE_CHANNELS),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    for l in range(1, layer):
        layers += [
            nn.Conv2d(BASE_CHANNELS, BASE_CHANNELS, kernel_size=3, stride=1, padding=2**l, dilation=2**l, bias=False),
            norm(BASE_CHANNELS),
            nn.LeakyReLU(0.2, inplace=True),
        ]
    layers += [
        nn.Conv2d(BASE_CHANNELS, BASE_CHANNELS, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(BASE_CHANNELS),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    backbone = nn.Sequential(*layers)
    tail = nn.Conv2d(BASE_CHANNELS, 1, kernel_size=1, stride=1, padding=0, dilation=1)
    backbone.apply(weights_init_identity)
    weights_init_identity(tail)
    return backbone, tail


class ParamAdapter(nn.Module):
    def __init__(self, param_dim=2, channels=BASE_CHANNELS, hidden=32, scale=0.1):
        super().__init__()
        self.scale = float(scale)
        self.fc1 = nn.Linear(param_dim, hidden)
        self.fc2 = nn.Linear(hidden, channels * 2)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, feat, params):
        h = F.relu(self.fc1(params), inplace=True)
        gamma_beta = self.fc2(h)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1) * self.scale
        beta = torch.tanh(beta).unsqueeze(-1).unsqueeze(-1) * self.scale
        return feat * (1.0 + gamma) + beta


class DeepGuidedFilterAdapter(nn.Module):
    def __init__(self, radius=1):
        super().__init__()
        self.lr_backbone, self.lr_tail = build_lr_backbone()
        self.adapter = ParamAdapter()
        self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm)

    def forward(self, x_lr, x_hr, params):
        params = normalize_wls_params(params).to(x_lr.dtype) - 0.5
        feat = self.lr_backbone(x_lr)
        feat = self.adapter(feat, params)
        y_lr = self.lr_tail(feat)
        return self.gf(x_lr, y_lr, x_hr).clamp(0, 1)


def build_model(variant):
    if variant == "paraminput_freezeb":
        return DeepGuidedFilterParamInput(radius=1)
    if variant == "paramtransfer_freezefront":
        return DeepGuidedFilterParamTransfer(radius=1)
    if variant == "paramadapter":
        return DeepGuidedFilterAdapter(radius=1)
    raise ValueError(f"Unsupported variant: {variant}")


def parse_wls_params(path):
    name = os.path.basename(path)
    match = re.search(r"-l([0-9eE+\-.]+)-a([0-9eE+\-.]+)-h", name)
    if match is None:
        raise ValueError(f"Cannot parse lambda/alpha from filename: {name}")
    return float(match.group(1)), float(match.group(2))


def normalize_wls_params(params):
    lambda_v = params[:, 0].clamp_min(1e-8)
    alpha_v = params[:, 1]
    log_lambda = torch.log10(lambda_v)
    log_min = math.log10(LAMBDA_MIN)
    log_max = math.log10(LAMBDA_MAX)
    lambda_norm = (log_lambda - log_min) / (log_max - log_min)
    alpha_norm = (alpha_v - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN)
    lambda_norm = lambda_norm.clamp(0.0, 1.0)
    alpha_norm = alpha_norm.clamp(0.0, 1.0)
    return torch.stack([lambda_norm, alpha_norm], dim=1)


class GuidedFilterDataset(Dataset):
    def __init__(self, x_lr_dir, x_hr_dir, label_dir):
        self.x_lr_files = sorted(
            [os.path.join(x_lr_dir, f) for f in os.listdir(x_lr_dir) if os.path.isfile(os.path.join(x_lr_dir, f))]
        )
        self.x_hr_files = sorted(
            [os.path.join(x_hr_dir, f) for f in os.listdir(x_hr_dir) if os.path.isfile(os.path.join(x_hr_dir, f))]
        )
        self.label_files = sorted(
            [os.path.join(label_dir, f) for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]
        )
        if len(self.x_lr_files) != len(self.x_hr_files) or len(self.x_hr_files) != len(self.label_files):
            raise ValueError(
                f"Dataset count mismatch: lr={len(self.x_lr_files)}, hr={len(self.x_hr_files)}, label={len(self.label_files)}"
            )

    def __len__(self):
        return len(self.x_hr_files)

    def __getitem__(self, idx):
        x_lr_path = self.x_lr_files[idx]
        x_hr_path = self.x_hr_files[idx]
        label_path = self.label_files[idx]
        x_lr = cv2.imread(x_lr_path, cv2.IMREAD_UNCHANGED).astype("float32")
        x_hr = cv2.imread(x_hr_path, cv2.IMREAD_UNCHANGED).astype("float32")
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED).astype("float32")
        lambda_val, alpha_val = parse_wls_params(label_path)
        return {
            "input_lr": torch.from_numpy(x_lr).unsqueeze(0).unsqueeze(0),
            "input_hr": torch.from_numpy(x_hr).unsqueeze(0).unsqueeze(0),
            "target_img": torch.from_numpy(label).unsqueeze(0).unsqueeze(0),
            "input_params": torch.tensor([lambda_val, alpha_val], dtype=torch.float32),
            "filename": os.path.splitext(os.path.basename(x_hr_path))[0],
            "input_height_hr": x_hr.shape[0],
            "input_width_hr": x_hr.shape[1],
            "input_height_lr": x_lr.shape[0],
            "input_width_lr": x_lr.shape[1],
        }


def var_custom_collate(batch):
    min_h_hr = min(item["input_height_hr"] for item in batch)
    min_w_hr = min(item["input_width_hr"] for item in batch)
    min_h_lr = min(item["input_height_lr"] for item in batch)
    min_w_lr = min(item["input_width_lr"] for item in batch)
    batch_x_hr = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_x_lr = torch.zeros(len(batch), 1, min_h_lr, min_w_lr)
    batch_target = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_params = torch.zeros(len(batch), 2)
    filenames = []
    for idx, item in enumerate(batch):
        batch_x_lr[idx] = item["input_lr"][:, :, :min_h_lr, :min_w_lr]
        batch_x_hr[idx] = item["input_hr"][:, :, :min_h_hr, :min_w_hr]
        batch_target[idx] = item["target_img"][:, :, :min_h_hr, :min_w_hr]
        batch_params[idx] = item["input_params"]
        filenames.append(item["filename"])
    return batch_x_lr, batch_x_hr, batch_target, batch_params, filenames


def crop_to_even(tensor):
    _, _, h, w = tensor.shape
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    return tensor[:, :, :new_h, :new_w]


def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100.0 if mse == 0 else 10.0 * np.log10(1.0 / mse)
    return psnr, mse


def save_float32_tiff(img, save_path):
    img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    imageio.imwrite(save_path, img.astype(np.float32))


def get_device(args):
    if args.cpu or not torch.cuda.is_available():
        return torch.device("cpu"), False
    visible_gpus = torch.cuda.device_count()
    if args.gpu < 0 or args.gpu >= visible_gpus:
        raise ValueError(f"--gpu={args.gpu} is invalid; visible CUDA devices: {visible_gpus}")
    return torch.device(f"cuda:{args.gpu}"), True


def clean_state_dict(checkpoint):
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if any(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def load_variant_model(variant, model_path, device):
    model = build_model(variant)
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = clean_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def evaluate_variant(args, variant, device, use_cuda, loader):
    model_name = MODEL_FILENAMES[variant][args.model_kind]
    model_path = os.path.join(args.model_dir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    model = load_variant_model(variant, model_path, device)
    pred_dir = os.path.join(args.output_root, f"{variant}_{args.model_kind}")
    if args.save_preds:
        os.makedirs(pred_dir, exist_ok=True)

    total_psnr = 0.0
    total_mse = 0.0
    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for batch_x_lr, batch_x_hr, batch_label, batch_params, filenames in tqdm(
            loader, desc=f"Eval {variant}", leave=False
        ):
            batch_x_hr = crop_to_even(batch_x_hr)
            batch_label = crop_to_even(batch_label)
            batch_x_lr = batch_x_lr.to(device)
            batch_x_hr = batch_x_hr.to(device)
            batch_label = batch_label.to(device)
            batch_params = batch_params.to(device)
            batch_x_lr = batch_x_lr - 0.5
            batch_x_hr = batch_x_hr - 0.5

            start_time = time.time()
            output = model(batch_x_lr, batch_x_hr, batch_params)
            if use_cuda:
                torch.cuda.synchronize(device)
            total_time += time.time() - start_time
            total_images += output.size(0)

            psnr, mse = calc_psnr(output, batch_label)
            total_psnr += psnr
            total_mse += mse

            if args.save_preds:
                for i, fname in enumerate(filenames):
                    save_float32_tiff(output[i], os.path.join(pred_dir, f"{fname}_pred.tiff"))

    result = {
        "variant": variant,
        "model_kind": args.model_kind,
        "model_path": model_path,
        "avg_mse": total_mse / len(loader),
        "avg_psnr": total_psnr / len(loader),
        "avg_ms_per_image": (total_time / max(total_images, 1)) * 1000.0,
        "num_samples": len(loader.dataset),
    }
    return result


def write_summary(summary_path, rows):
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    fieldnames = ["variant", "model_kind", "avg_mse", "avg_psnr", "avg_ms_per_image", "num_samples", "model_path"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    seed_everything(42)
    device, use_cuda = get_device(args)
    dataset = GuidedFilterDataset(args.x_lr_dir, args.x_hr_dir, args.label_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=var_custom_collate,
        pin_memory=use_cuda,
        drop_last=False,
    )

    results = []
    for variant in args.variants:
        result = evaluate_variant(args, variant, device, use_cuda, loader)
        results.append(result)
        print(
            f"[RESULT] variant={result['variant']} kind={result['model_kind']} "
            f"mse={result['avg_mse']:.6f} psnr={result['avg_psnr']:.4f} "
            f"ms_per_image={result['avg_ms_per_image']:.2f} samples={result['num_samples']}"
        )

    write_summary(args.summary_path, results)
    print(f"Summary saved to: {args.summary_path}")


if __name__ == "__main__":
    main()
