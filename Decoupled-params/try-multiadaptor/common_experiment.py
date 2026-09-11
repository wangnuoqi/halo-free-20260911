import argparse
import datetime
import math
import os
import platform
import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

LAMBDA_MIN = 0.025
LAMBDA_MAX = 2.5
ALPHA_MIN = 0.8
ALPHA_MAX = 1.6
BASE_CHANNELS = 24


def _detect_workspace_root():
    env_root = os.environ.get("NUOQI_WORKSPACE")
    if env_root:
        return Path(env_root)
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "halo-free").exists() and (parent / "S2R-HDR").exists():
            return parent
    if platform.system() == "Windows":
        return Path("D:/nuoqi")
    return Path("/mnt/d/nuoqi")


WORKSPACE_ROOT = _detect_workspace_root()
TRY_MULTIADAPTOR_ROOT = WORKSPACE_ROOT / "halo-free" / "Decoupled-params" / "try-multiadaptor"
PRETRAINED_PATH = str(
    WORKSPACE_ROOT / "halo-free" / "Decoupled_Learning" / "HDRv1" / "models_retrain" / "convguided_filter_best_mix.pth"
)
TRAIN_X_LR_DIR = str(WORKSPACE_ROOT / "S2R-HDR" / "data" / "mixed-lavel-singlehdr-param" / "traininput_downsampled")
TRAIN_X_HR_DIR = str(WORKSPACE_ROOT / "S2R-HDR" / "data" / "mixed-lavel-singlehdr-param" / "traininput")
TRAIN_LABEL_DIR = str(WORKSPACE_ROOT / "S2R-HDR" / "data" / "mixed-lavel-singlehdr-param" / "trainlabel")
TEST_X_LR_DIR = str(WORKSPACE_ROOT / "S2R-HDR" / "data" / "mixed-lavel-singlehdr-param" / "testinput_downsampled")
TEST_X_HR_DIR = str(WORKSPACE_ROOT / "S2R-HDR" / "data" / "mixed-lavel-singlehdr-param" / "testinput")
TEST_LABEL_DIR = str(WORKSPACE_ROOT / "S2R-HDR" / "data" / "mixed-lavel-singlehdr-param" / "testlabel")


@dataclass
class ExperimentConfig:
    name: str
    description: str
    visible_devices: str
    cond_channels: int = 8
    adapter_hidden: int = 12
    adapter_scale: float = 0.2
    adapter_bottleneck: int = 8
    freeze_mode: str = "multi_adapter_b"
    adapter_positions: tuple = (2, 5, 8, 11, 14, 17)


def build_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train-batch-per-gpu", type=int, default=12, help="Per-GPU batch size for training.")
    parser.add_argument("--test-batch", type=int, default=4, help="Batch size used during validation.")
    parser.add_argument("--train-workers", type=int, default=12, help="Number of dataloader workers for training.")
    parser.add_argument("--test-workers", type=int, default=2, help="Number of dataloader workers for validation.")
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory.")
    parser.add_argument("--epochs", type=int, default=80, help="Total number of epochs to train.")
    parser.add_argument("--save-dir", default=None, help="Override checkpoint output directory.")
    parser.add_argument("--resume-path", default=None, help="Override resume checkpoint path.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from the default full checkpoint.")
    return parser


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


class ParamMapEncoder(nn.Module):
    def __init__(self, param_dim=2, hidden=16, out_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(param_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, params):
        return self.fc2(F.relu(self.fc1(params), inplace=True))


class ConditionalResidualAdapter(nn.Module):
    def __init__(self, param_dim=2, channels=BASE_CHANNELS, hidden=12, bottleneck=8, scale=0.2):
        super().__init__()
        self.scale = float(scale)
        self.down = nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False)
        self.up = nn.Conv2d(bottleneck, channels, kernel_size=1, bias=False)
        self.fc1 = nn.Linear(param_dim, hidden)
        self.fc2 = nn.Linear(hidden, channels * 2)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, feat, params):
        h = F.relu(self.fc1(params), inplace=True)
        gamma_beta = self.fc2(h)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1) * self.scale
        beta = torch.tanh(beta).unsqueeze(-1).unsqueeze(-1) * self.scale
        delta = self.up(F.relu(self.down(feat), inplace=True))
        delta = delta * (1.0 + gamma) + beta
        return feat + delta


def build_lr_layers(in_channels, norm=AdaptiveNorm, layer=5):
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


class DeepGuidedFilterMultiAdapter(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        in_channels = 1 + config.cond_channels
        self.lr = build_lr_layers(in_channels=in_channels)
        self.gf = ConvGuidedFilter(radius=1, norm=AdaptiveNorm)
        self.param_encoder = ParamMapEncoder(param_dim=2, hidden=16, out_dim=config.cond_channels)
        self.adapters = nn.ModuleDict(
            {
                str(idx): ConditionalResidualAdapter(
                    param_dim=2,
                    channels=BASE_CHANNELS,
                    hidden=config.adapter_hidden,
                    bottleneck=config.adapter_bottleneck,
                    scale=config.adapter_scale,
                )
                for idx in config.adapter_positions
            }
        )

    def build_condition_maps(self, x_lr, params):
        norm_params = normalize_wls_params(params).to(x_lr.dtype) - 0.5
        cond = self.param_encoder(norm_params)
        return cond[:, :, None, None].expand(-1, -1, x_lr.size(2), x_lr.size(3)), norm_params

    def forward(self, x_lr, x_hr, params):
        param_maps, norm_params = self.build_condition_maps(x_lr, params)
        feat = torch.cat([x_lr, param_maps], dim=1)
        for idx, layer in enumerate(self.lr):
            feat = layer(feat)
            adapter_key = str(idx)
            if adapter_key in self.adapters:
                feat = self.adapters[adapter_key](feat, norm_params)
        return self.gf(x_lr, feat, x_hr).clamp(0, 1)


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
        self.x_lr_files = sorted([os.path.join(x_lr_dir, f) for f in os.listdir(x_lr_dir)])
        self.x_hr_files = sorted([os.path.join(x_hr_dir, f) for f in os.listdir(x_hr_dir)])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        print(f"LR: {len(self.x_lr_files)}, HR: {len(self.x_hr_files)}, Label: {len(self.label_files)}")

    def __len__(self):
        return min(len(self.x_lr_files), len(self.x_hr_files), len(self.label_files))

    def __getitem__(self, idx):
        x_lr = cv2.imread(self.x_lr_files[idx], cv2.IMREAD_UNCHANGED).astype("float32")
        x_hr = cv2.imread(self.x_hr_files[idx], cv2.IMREAD_UNCHANGED).astype("float32")
        label = cv2.imread(self.label_files[idx], cv2.IMREAD_UNCHANGED).astype("float32")
        lambda_val, alpha_val = parse_wls_params(self.label_files[idx])
        input_hr_h, input_hr_w = x_hr.shape
        input_lr_h, input_lr_w = x_lr.shape
        return {
            "input_lr": torch.from_numpy(x_lr).unsqueeze(0).unsqueeze(0),
            "input_hr": torch.from_numpy(x_hr).unsqueeze(0).unsqueeze(0),
            "target_img": torch.from_numpy(label).unsqueeze(0).unsqueeze(0),
            "input_params": torch.tensor([lambda_val, alpha_val], dtype=torch.float32),
            "input_height_hr": input_hr_h,
            "input_width_hr": input_hr_w,
            "input_height_lr": input_lr_h,
            "input_width_lr": input_lr_w,
        }


def var_custom_collate(batch):
    min_h_hr = min(item["input_height_hr"] for item in batch)
    min_w_hr = min(item["input_width_hr"] for item in batch)
    min_h_lr = min(item["input_height_lr"] for item in batch)
    min_w_lr = min(item["input_width_lr"] for item in batch)
    batch_x_hr = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_x_lr = torch.zeros(len(batch), 1, min_h_lr, min_w_lr)
    batch_target_images = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_params = torch.zeros(len(batch), 2)
    for idx, item in enumerate(batch):
        batch_x_lr[idx] = item["input_lr"][:, :, :min_h_lr, :min_w_lr]
        batch_x_hr[idx] = item["input_hr"][:, :, :min_h_hr, :min_w_hr]
        batch_target_images[idx] = item["target_img"][:, :, :min_h_hr, :min_w_hr]
        batch_params[idx] = item["input_params"]
    return batch_x_lr, batch_x_hr, batch_target_images, batch_params


def crop_to_even(tensor):
    _, _, h, w = tensor.shape
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    return tensor[:, :, :new_h, :new_w]


def _to_gray(x):
    if x.dim() == 3:
        x = x.unsqueeze(1)
    return x if x.size(1) == 1 else x.mean(dim=1, keepdim=True)


def _sobel_grad(x, eps=1e-6):
    x = F.pad(x, (1, 1, 1, 1), mode="reflect")
    kx = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]], device=x.device, dtype=x.dtype)
    ky = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]], device=x.device, dtype=x.dtype)
    gx = F.conv2d(x, kx)
    gy = F.conv2d(x, ky)
    mag = torch.sqrt(gx * gx + gy * gy + eps)
    return gx, gy, mag


def sobel_gradient_loss(pred, target, mode="vector", reduction="mean", eps=1e-6):
    pred_g = _to_gray(pred)
    targ_g = _to_gray(target)
    pgx, pgy, pm = _sobel_grad(pred_g, eps)
    tgx, tgy, tm = _sobel_grad(targ_g, eps)
    if mode == "vector":
        loss = (pgx - tgx).abs() + (pgy - tgy).abs()
    elif mode == "magnitude":
        loss = (pm - tm).abs()
    else:
        raise ValueError("mode must be 'vector' or 'magnitude'")
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be 'mean' | 'sum' | 'none'")


def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1 / mse)
    return psnr, mse


def _unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def set_requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad = requires_grad


def load_pretrained_single_param(model, path):
    if path is None or not os.path.exists(path):
        print(f"=> Pretrained checkpoint not found: {path}")
        return False
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    current_state = model.state_dict()
    adapted_state = {}
    for key, value in current_state.items():
        source_value = None
        if key in state_dict:
            source_value = state_dict[key]
        elif key.startswith("module.") and key[len("module."):] in state_dict:
            source_value = state_dict[key[len("module."):]]
        elif (not key.startswith("module.")) and f"module.{key}" in state_dict:
            source_value = state_dict[f"module.{key}"]
        if source_value is None:
            adapted_state[key] = value
            continue
        stripped_key = key[len("module."):] if key.startswith("module.") else key
        if stripped_key == "lr.0.weight" and source_value.shape[1] == 1 and value.shape[1] > 1:
            new_weight = value.clone()
            new_weight[:, :1, :, :] = source_value
            new_weight[:, 1:, :, :] = 0.0
            adapted_state[key] = new_weight
        elif source_value.shape == value.shape:
            adapted_state[key] = source_value
        else:
            adapted_state[key] = value
    model.load_state_dict(adapted_state, strict=True)
    print(f"=> Loaded pretrained weights from: {path}")
    return True


def freeze_model(model, config: ExperimentConfig):
    net = _unwrap_model(model)
    set_requires_grad(net.lr, False)
    set_requires_grad(net.gf, False)
    set_requires_grad(net.param_encoder, True)
    set_requires_grad(net.adapters, True)
    train_indices = [0, 1, 2, 15, 16, 17, 18]
    if config.freeze_mode != "multi_adapter_b":
        raise ValueError(f"Unknown freeze_mode: {config.freeze_mode}")
    for idx in train_indices:
        if idx < len(net.lr):
            set_requires_grad(net.lr[idx], True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"=> Freeze policy {config.freeze_mode} active: trainable {trainable}/{total} params")


def run_experiment(args, config: ExperimentConfig):
    seed_everything(42)
    base_dir = os.path.join(str(TRY_MULTIADAPTOR_ROOT), config.name)
    save_dir = args.save_dir or os.path.join(base_dir, "model")
    log_root = os.path.join(base_dir, "log_dir")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    device_ids = list(range(torch.cuda.device_count()))
    if not device_ids:
        raise RuntimeError("No CUDA devices are visible. Please set CUDA_VISIBLE_DEVICES correctly.")
    device = torch.device("cuda:0")

    model = DeepGuidedFilterMultiAdapter(config).to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0]).to(device)
    print(f"=> Using DataParallel device_ids={device_ids}")
    load_pretrained_single_param(model, PRETRAINED_PATH)
    freeze_model(model, config)

    train_dataset = GuidedFilterDataset(TRAIN_X_LR_DIR, TRAIN_X_HR_DIR, TRAIN_LABEL_DIR)
    test_dataset = GuidedFilterDataset(TEST_X_LR_DIR, TEST_X_HR_DIR, TEST_LABEL_DIR)
    train_batch_size = max(1, args.train_batch_per_gpu)
    test_batch_size = max(1, args.test_batch)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=args.train_workers,
        collate_fn=var_custom_collate,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=args.test_workers,
        collate_fn=var_custom_collate,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    criterion = torch.nn.MSELoss(reduction="mean")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(log_root, f"{config.name}_{timestamp}"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    best_loss = float("inf")
    start_epoch = 0
    full_name = f"convguided_filter_full_{config.name}.pth"
    last_name = f"convguided_filter_last_{config.name}.pth"
    best_name = f"convguided_filter_best_{config.name}.pth"
    resume_path = args.resume_path or os.path.join(save_dir, full_name)

    if (not args.no_resume) and os.path.exists(resume_path):
        print(f"=> Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"=> Resumed from epoch {start_epoch}, best_loss = {best_loss:.6f}")
    else:
        print("=> No checkpoint found, training from scratch.")

    global_train_batch = train_batch_size * len(device_ids)
    print(f"=> Batch config: per_gpu_train={train_batch_size}, global_train={global_train_batch}, test_batch={test_batch_size}")

    def save_checkpoint(epoch, best_loss_value, is_best=False):
        full_path = os.path.join(save_dir, full_name)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss_value,
            },
            full_path,
        )
        print(f"Checkpoint saved: {full_path}")
        last_path = os.path.join(save_dir, last_name)
        torch.save(model.state_dict(), last_path)
        print(f"Last model weights saved: {last_path}")
        if is_best:
            best_path = os.path.join(save_dir, best_name)
            torch.save(model.state_dict(), best_path)
            print(f"*** Best model updated: {best_path} (Test MSE: {best_loss_value:.6f}) ***")

    for epoch in tqdm(range(start_epoch, args.epochs), desc="Training Progress", dynamic_ncols=True):
        model.train()
        epoch_mse = 0.0
        epoch_psnr = 0.0
        for batch_idx, (batch_x_lr, batch_x_hr, batch_label, batch_params) in enumerate(train_loader):
            batch_x_hr = crop_to_even(batch_x_hr)
            batch_label = crop_to_even(batch_label)
            batch_x_lr, batch_x_hr, batch_label, batch_params = (
                batch_x_lr.to(device),
                batch_x_hr.to(device),
                batch_label.to(device),
                batch_params.to(device),
            )
            batch_x_lr = batch_x_lr - 0.5
            batch_x_hr = batch_x_hr - 0.5
            output = model(batch_x_lr, batch_x_hr, batch_params)
            eps = 1e-8
            loss_mse = criterion(output, batch_label)
            loss_g = sobel_gradient_loss(output, batch_label, mode="vector")
            scale_g = (loss_g / (loss_mse + eps)).detach().clamp_min(eps)
            loss = loss_mse + loss_g / scale_g
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr, mse = calc_psnr(output.detach(), batch_label)
            if batch_idx % 200 == 0:
                print(f"iter {batch_idx} MSE loss: {mse:.6f}")
            epoch_mse += mse
            epoch_psnr += psnr
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 50 == 0:
                writer.add_scalar("train/mse", mse, global_step)
                writer.add_scalar("train/psnr", psnr, global_step)

        avg_epoch_mse = epoch_mse / len(train_loader)
        avg_epoch_psnr = epoch_psnr / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs} - MSE: {avg_epoch_mse:.6f} - PSNR: {avg_epoch_psnr:.2f} dB")
        writer.add_scalar("train/mse_epoch", avg_epoch_mse, epoch)
        writer.add_scalar("train/psnr_epoch", avg_epoch_psnr, epoch)
        scheduler.step()

        model.eval()
        test_mse = 0.0
        test_psnr = 0.0
        with torch.no_grad():
            for test_x_lr, test_x_hr, test_label, test_params in test_loader:
                test_x_hr = crop_to_even(test_x_hr)
                test_label = crop_to_even(test_label)
                test_x_lr, test_x_hr, test_label, test_params = (
                    test_x_lr.to(device),
                    test_x_hr.to(device),
                    test_label.to(device),
                    test_params.to(device),
                )
                test_x_lr = test_x_lr - 0.5
                test_x_hr = test_x_hr - 0.5
                test_output = model(test_x_lr, test_x_hr, test_params)
                psnr, mse = calc_psnr(test_output, test_label)
                test_mse += mse
                test_psnr += psnr

        avg_test_mse = test_mse / len(test_loader)
        avg_test_psnr = test_psnr / len(test_loader)
        print(f"Epoch {epoch + 1}/{args.epochs} - Test MSE: {avg_test_mse:.6f} - Test PSNR: {avg_test_psnr:.2f} dB")
        writer.add_scalar("test/mse", avg_test_mse, epoch)
        writer.add_scalar("test/psnr", avg_test_psnr, epoch)
        is_best = avg_test_mse < best_loss
        if is_best:
            best_loss = avg_test_mse
        save_checkpoint(epoch, best_loss, is_best=is_best)
    writer.close()
