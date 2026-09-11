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
TRY_PARAMADAPTER_ROOT = WORKSPACE_ROOT / "halo-free" / "Decoupled-params" / "try-paramadapter"
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
    train_mode: str = "full"
    progressive_unfreeze_epochs: tuple = ()
    base_channels: int = 64
    inject_channels: int = 64


def build_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train-batch-per-gpu", type=int, default=12, help="Per-GPU batch size for training.")
    parser.add_argument("--test-batch", type=int, default=4, help="Batch size used during validation.")
    parser.add_argument("--train-workers", type=int, default=8, help="Training dataloader workers.")
    parser.add_argument("--test-workers", type=int, default=8, help="Validation dataloader workers.")
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
        for i in range(min(n_in, n_out)):
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


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels, dilation, norm=AdaptiveNorm):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.norm1 = norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.norm2 = norm(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class DeepFeatureModulator(nn.Module):
    def __init__(self, param_dim=2, hidden=16, channels=64):
        super().__init__()
        self.fc1_weight = nn.Linear(param_dim, hidden)
        self.fc1_bias = nn.Linear(param_dim, hidden)
        self.fc2_weight = nn.Linear(hidden, channels)
        self.fc2_bias = nn.Linear(hidden, channels)
        self.param_relu = nn.ReLU(inplace=True)

    def forward(self, feat, params):
        norm_weight = self.fc2_weight(self.param_relu(self.fc1_weight(params)))
        norm_bias = self.fc2_bias(self.param_relu(self.fc1_bias(params)))
        batch_outputs = []
        for n in range(feat.size(0)):
            feat_n = F.instance_norm(feat.narrow(0, n, 1), weight=norm_weight[n], bias=norm_bias[n])
            batch_outputs.append(F.leaky_relu(feat_n, negative_slope=0.2, inplace=False))
        return torch.cat(batch_outputs, dim=0)


class LRNetDeepAdapter(nn.Module):
    def __init__(self, base_channels=64, inject_channels=64, norm=AdaptiveNorm):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            norm(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        dilations = [1, 2, 4, 8, 4, 2]
        self.body = nn.Sequential(*[ResidualDilatedBlock(base_channels, dilation=d, norm=norm) for d in dilations])
        self.pre_inject = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            norm(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.inject_conv = nn.Conv2d(base_channels, inject_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.modulator = DeepFeatureModulator(param_dim=2, hidden=16, channels=inject_channels)
        self.out_conv = nn.Conv2d(inject_channels, 1, kernel_size=1, stride=1, padding=0, dilation=1)
        self.apply(weights_init_identity)

    def forward(self, x, params):
        feat = self.head(x)
        feat = self.body(feat)
        feat = self.pre_inject(feat)
        feat = self.inject_conv(feat)
        feat = self.modulator(feat, params)
        return self.out_conv(feat)


class DeepGuidedFilterParamAdapter(nn.Module):
    def __init__(self, config: ExperimentConfig, radius=1):
        super().__init__()
        self.lr = LRNetDeepAdapter(base_channels=config.base_channels, inject_channels=config.inject_channels)
        self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm)

    def forward(self, x_lr, x_hr, params):
        params = normalize_wls_params(params).to(x_lr.dtype) - 0.5
        return self.gf(x_lr, self.lr(x_lr, params), x_hr).clamp(0, 1)


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
        return {
            "input_lr": torch.from_numpy(x_lr).unsqueeze(0).unsqueeze(0),
            "input_hr": torch.from_numpy(x_hr).unsqueeze(0).unsqueeze(0),
            "target_img": torch.from_numpy(label).unsqueeze(0).unsqueeze(0),
            "input_params": torch.tensor([lambda_val, alpha_val], dtype=torch.float32),
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


def get_stage_for_epoch(epoch, config: ExperimentConfig):
    if config.train_mode != "progressive":
        return 0
    boundaries = tuple(config.progressive_unfreeze_epochs or ())
    if len(boundaries) != 2:
        raise ValueError("progressive mode expects exactly 2 stage boundary epochs.")
    if epoch < boundaries[0]:
        return 0
    if epoch < boundaries[1]:
        return 1
    return 2


def apply_training_stage(model, config: ExperimentConfig, stage_idx: int):
    net = _unwrap_model(model)
    if config.train_mode == "full":
        set_requires_grad(net.lr, True)
        set_requires_grad(net.gf, True)
        stage_name = "full_train"
    elif config.train_mode == "progressive":
        set_requires_grad(net.lr, False)
        set_requires_grad(net.gf, False)
        set_requires_grad(net.lr.inject_conv, True)
        set_requires_grad(net.lr.modulator, True)
        set_requires_grad(net.lr.out_conv, True)
        if stage_idx >= 1:
            set_requires_grad(net.lr.head, True)
            set_requires_grad(net.lr.body, True)
            set_requires_grad(net.lr.pre_inject, True)
        if stage_idx >= 2:
            set_requires_grad(net.gf, True)
        stage_name = {0: "stage0_modulator_only", 1: "stage1_full_lr", 2: "stage2_full_model"}[stage_idx]
    else:
        raise ValueError(f"Unknown train_mode: {config.train_mode}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"=> Training policy active: {stage_name} trainable {trainable}/{total} params")


def run_experiment(args, config: ExperimentConfig):
    seed_everything(42)
    base_dir = os.path.join(str(TRY_PARAMADAPTER_ROOT), config.name)
    save_dir = args.save_dir or os.path.join(base_dir, "model")
    log_root = os.path.join(base_dir, "log_dir")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    device_ids = list(range(torch.cuda.device_count()))
    if not device_ids:
        raise RuntimeError("No CUDA devices are visible. Please set CUDA_VISIBLE_DEVICES correctly.")
    device = torch.device("cuda:0")

    model = DeepGuidedFilterParamAdapter(config).to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0]).to(device)
    print(f"=> Using DataParallel device_ids={device_ids}")
    current_stage = get_stage_for_epoch(0, config)
    apply_training_stage(model, config, current_stage)

    train_dataset = GuidedFilterDataset(TRAIN_X_LR_DIR, TRAIN_X_HR_DIR, TRAIN_LABEL_DIR)
    test_dataset = GuidedFilterDataset(TEST_X_LR_DIR, TEST_X_HR_DIR, TEST_LABEL_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, args.train_batch_per_gpu),
        shuffle=True,
        num_workers=args.train_workers,
        collate_fn=var_custom_collate,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(1, args.test_batch),
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
        current_stage = get_stage_for_epoch(start_epoch, config)
        apply_training_stage(model, config, current_stage)
        print(f"=> Resumed from epoch {start_epoch}, best_loss = {best_loss:.6f}")
    else:
        print("=> No checkpoint found, training from scratch.")

    print(f"=> Batch config: per_gpu_train={max(1, args.train_batch_per_gpu)}, global_train={max(1, args.train_batch_per_gpu) * len(device_ids)}, test_batch={max(1, args.test_batch)}")

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
        stage_for_epoch = get_stage_for_epoch(epoch, config)
        if stage_for_epoch != current_stage:
            current_stage = stage_for_epoch
            apply_training_stage(model, config, current_stage)
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
            loss_mse = criterion(output, batch_label)
            loss_g = sobel_gradient_loss(output, batch_label, mode="vector")
            scale_g = (loss_g / (loss_mse + 1e-8)).detach().clamp_min(1e-8)
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
