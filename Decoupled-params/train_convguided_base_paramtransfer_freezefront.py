import argparse
import datetime
import math
import os
import random
import re

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

LAMBDA_MIN = 0.025
LAMBDA_MAX = 2.5
ALPHA_MIN = 0.8
ALPHA_MAX = 1.6
BASE_CHANNELS = 24
PRETRAINED_PATH = "/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/models_retrain/convguided_filter_best_mix.pth"

downsample_folder_train = "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/traininput_downsampled"
input_folder_train = "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/traininput"
label_train = "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/trainlabel"

downsample_folder_test = "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/testinput_downsampled"
input_folder_test = "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/testinput"
label_test = "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/testlabel"


def parse_args():
    parser = argparse.ArgumentParser(
        description="DataParallel training for pretrained single-param ConvGuidedFilter with frozen front-half and late param modulation."
    )
    parser.add_argument("--save-dir", default="/mnt/d/nuoqi/halo-free/Decoupled-params/model")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--resume-path", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--train-batch-per-gpu", type=int, default=24, help="Per-GPU batch size for training.")
    parser.add_argument("--test-batch", type=int, default=8, help="Batch size used by rank0 during validation.")
    parser.add_argument("--train-workers", type=int, default=24, help="Number of dataloader workers per rank for training.")
    parser.add_argument("--test-workers", type=int, default=12, help="Number of dataloader workers used by rank0 for validation.")
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory.")
    parser.add_argument("--unfreeze-gf-conv-a", action="store_true")
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


def load_pretrained_single_param(model, path):
    if path is None or not os.path.exists(path):
        print(f"=> Pretrained checkpoint not found: {path}")
        return False

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cleaned = {k[len("module."): ] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    net = _unwrap_model(model)

    current_front = net.lr_front.state_dict()
    current_back = net.lr_back.state_dict()
    current_out = net.lr_out.state_dict()
    current_gf = net.gf.state_dict()

    front_src = {}
    for idx in range(9):
        src_key = f"lr.{idx}"
        if f"{src_key}.weight" in cleaned:
            front_src[f"{idx}.weight"] = cleaned[f"{src_key}.weight"]
        if f"{src_key}.bias" in cleaned and f"{idx}.bias" in current_front:
            front_src[f"{idx}.bias"] = cleaned[f"{src_key}.bias"]
        if f"{src_key}.w_0" in cleaned:
            front_src[f"{idx}.w_0"] = cleaned[f"{src_key}.w_0"]
        if f"{src_key}.w_1" in cleaned:
            front_src[f"{idx}.w_1"] = cleaned[f"{src_key}.w_1"]
        if f"{src_key}.bn.weight" in cleaned:
            front_src[f"{idx}.bn.weight"] = cleaned[f"{src_key}.bn.weight"]
            front_src[f"{idx}.bn.bias"] = cleaned[f"{src_key}.bn.bias"]
            front_src[f"{idx}.bn.running_mean"] = cleaned[f"{src_key}.bn.running_mean"]
            front_src[f"{idx}.bn.running_var"] = cleaned[f"{src_key}.bn.running_var"]
            front_src[f"{idx}.bn.num_batches_tracked"] = cleaned[f"{src_key}.bn.num_batches_tracked"]
    current_front.update(front_src)
    net.lr_front.load_state_dict(current_front)

    back_src = {}
    for dst_idx, src_idx in enumerate(range(9, 18)):
        src_key = f"lr.{src_idx}"
        if f"{src_key}.weight" in cleaned:
            back_src[f"{dst_idx}.weight"] = cleaned[f"{src_key}.weight"]
        if f"{src_key}.bias" in cleaned and f"{dst_idx}.bias" in current_back:
            back_src[f"{dst_idx}.bias"] = cleaned[f"{src_key}.bias"]
        if f"{src_key}.w_0" in cleaned:
            back_src[f"{dst_idx}.w_0"] = cleaned[f"{src_key}.w_0"]
        if f"{src_key}.w_1" in cleaned:
            back_src[f"{dst_idx}.w_1"] = cleaned[f"{src_key}.w_1"]
        if f"{src_key}.bn.weight" in cleaned:
            back_src[f"{dst_idx}.bn.weight"] = cleaned[f"{src_key}.bn.weight"]
            back_src[f"{dst_idx}.bn.bias"] = cleaned[f"{src_key}.bn.bias"]
            back_src[f"{dst_idx}.bn.running_mean"] = cleaned[f"{src_key}.bn.running_mean"]
            back_src[f"{dst_idx}.bn.running_var"] = cleaned[f"{src_key}.bn.running_var"]
            back_src[f"{dst_idx}.bn.num_batches_tracked"] = cleaned[f"{src_key}.bn.num_batches_tracked"]
    current_back.update(back_src)
    net.lr_back.load_state_dict(current_back)

    if "lr.18.weight" in cleaned:
        current_out["weight"] = cleaned["lr.18.weight"]
    if "lr.18.bias" in cleaned:
        current_out["bias"] = cleaned["lr.18.bias"]
    net.lr_out.load_state_dict(current_out)

    gf_src = {k: cleaned[k] for k in current_gf if k in cleaned and cleaned[k].shape == current_gf[k].shape}
    current_gf.update(gf_src)
    net.gf.load_state_dict(current_gf)
    print(f"=> Loaded pretrained single-param weights from: {path}")
    return True


def freeze_policy(model, unfreeze_gf_conv_a=False):
    net = _unwrap_model(model)
    set_requires_grad(net.lr_front, False)
    set_requires_grad(net.gf, False)
    set_requires_grad(net.lr_back, True)
    set_requires_grad(net.param_mod, True)
    set_requires_grad(net.lr_out, True)
    if unfreeze_gf_conv_a:
        set_requires_grad(net.gf.conv_a, True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"=> Freeze policy active: trainable {trainable}/{total} params")
    print(f"=> lr_front frozen, gf frozen={not unfreeze_gf_conv_a}, lr_back/param_mod/lr_out trainable")


def main():
    args = parse_args()
    seed_everything(42)
    device_ids = [0, 1]
    device = torch.device("cuda:0")
    visible_gpus = torch.cuda.device_count()
    if visible_gpus <= max(device_ids):
        raise RuntimeError(
            f"Requested GPUs {device_ids}, but only {visible_gpus} CUDA device(s) are visible. "
            "Please set CUDA_VISIBLE_DEVICES correctly."
        )

    model = DeepGuidedFilterParamTransfer(radius=1).to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0]).to(device)
    print(f"=> Using DataParallel device_ids={device_ids}")
    load_pretrained_single_param(model, PRETRAINED_PATH)
    freeze_policy(model, unfreeze_gf_conv_a=args.unfreeze_gf_conv_a)

    train_dataset = GuidedFilterDataset(downsample_folder_train, input_folder_train, label_train)
    test_dataset = GuidedFilterDataset(downsample_folder_test, input_folder_test, label_test)
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
    task_name = "convguided_filter_mix_paramtransfer_freezefront"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("/mnt/d/nuoqi/halo-free/Decoupled-params/log_dir", f"{task_name}_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    best_loss = float("inf")
    start_epoch = 0
    os.makedirs(args.save_dir, exist_ok=True)

    resume_path = args.resume_path or os.path.join(args.save_dir, "convguided_filter_full_mix_paramtransfer_freezefront.pth")
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

    def save_checkpoint(epoch, best_loss_value, is_best=False):
        full_path = os.path.join(args.save_dir, "convguided_filter_full_mix_paramtransfer_freezefront.pth")
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

        last_model_path = os.path.join(args.save_dir, "convguided_filter_last_mix_paramtransfer_freezefront.pth")
        torch.save(model.state_dict(), last_model_path)
        print(f"Last model weights saved: {last_model_path}")

        if is_best:
            best_model_path = os.path.join(args.save_dir, "convguided_filter_best_mix_paramtransfer_freezefront.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"*** Best model updated: {best_model_path} (Test MSE: {best_loss_value:.6f}) ***")

    global_train_batch = train_batch_size * len(device_ids)
    print(
        f"=> Batch config: per_gpu_train={train_batch_size}, "
        f"global_train={global_train_batch}, test_batch={test_batch_size}"
    )

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
            loss_mse = criterion(output, batch_label)
            loss_g = sobel_gradient_loss(output, batch_label, mode="vector")
            scale_g = (loss_g / (loss_mse + 1e-8)).detach().clamp_min(1e-8)
            loss = loss_mse + loss_g / scale_g

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr, mse = calc_psnr(output.detach(), batch_label)
            epoch_mse += mse
            epoch_psnr += psnr
            if batch_idx % 200 == 0:
                print(f"iter {batch_idx} MSE loss: {mse:.6f}")

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


if __name__ == "__main__":
    main()
