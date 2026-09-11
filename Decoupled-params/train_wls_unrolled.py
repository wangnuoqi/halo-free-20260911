import argparse
import datetime
import os
import random
import re

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a learned WLS-style unrolled solver.")
    parser.add_argument(
        "--save-dir",
        default="/mnt/d/nuoqi/halo-free/Decoupled-params/model",
        help="Directory for checkpoints and model weights.",
    )
    parser.add_argument(
        "--log-root",
        default="/mnt/d/nuoqi/halo-free/Decoupled-params/log_dir",
        help="TensorBoard log root.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--test-num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=5, help="Number of unrolled WLS refinement steps.")
    parser.add_argument(
        "--resume-path",
        default=None,
        help="Full checkpoint path to resume from. Defaults to learned_wls_full.pth under save-dir.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch even if a checkpoint exists.",
    )
    return parser.parse_args()


args = parse_args()


def parse_wls_params(path):
    name = os.path.basename(path)
    match = re.search(r"-l([0-9eE+\-.]+)-a([0-9eE+\-.]+)-h", name)
    if match is None:
        raise ValueError(f"Cannot parse lambda/alpha from filename: {name}")
    return float(match.group(1)), float(match.group(2))


def crop_to_even(tensor):
    _, _, h, w = tensor.shape
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    return tensor[:, :, :new_h, :new_w]


def _to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.size(1) == 1:
        return x
    return x.mean(dim=1, keepdim=True)


def _sobel_grad(x: torch.Tensor, eps: float = 1e-6):
    x = F.pad(x, (1, 1, 1, 1), mode="reflect")
    kx = torch.tensor(
        [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]],
        device=x.device,
        dtype=x.dtype,
    )
    ky = torch.tensor(
        [[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]],
        device=x.device,
        dtype=x.dtype,
    )
    gx = F.conv2d(x, kx)
    gy = F.conv2d(x, ky)
    mag = torch.sqrt(gx * gx + gy * gy + eps)
    return gx, gy, mag


def sobel_gradient_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_g = _to_gray(pred)
    targ_g = _to_gray(target)
    pgx, pgy, _ = _sobel_grad(pred_g, eps)
    tgx, tgy, _ = _sobel_grad(targ_g, eps)
    return ((pgx - tgx).abs() + (pgy - tgy).abs()).mean()


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
    min_h_hr = min([item["input_height_hr"] for item in batch])
    min_w_hr = min([item["input_width_hr"] for item in batch])
    min_h_lr = min([item["input_height_lr"] for item in batch])
    min_w_lr = min([item["input_width_lr"] for item in batch])

    batch_x_hr = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_x_lr = torch.zeros(len(batch), 1, min_h_lr, min_w_lr)
    batch_target_images = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_params = torch.zeros(len(batch), 2)

    for idx, item in enumerate(batch):
        x_lr_crop = item["input_lr"][:, :, :min_h_lr, :min_w_lr]
        x_hr_crop = item["input_hr"][:, :, :min_h_hr, :min_w_hr]
        label_crop = item["target_img"][:, :, :min_h_hr, :min_w_hr]

        batch_x_lr[idx] = x_lr_crop
        batch_x_hr[idx] = x_hr_crop
        batch_target_images[idx] = label_crop
        batch_params[idx] = item["input_params"]

    return batch_x_lr, batch_x_hr, batch_target_images, batch_params


class AdaptiveNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.w_0 = nn.Parameter(torch.tensor([1.0]))
        self.w_1 = nn.Parameter(torch.tensor([0.0]))
        self.bn = nn.BatchNorm2d(channels, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


class ParamConditioning(nn.Module):
    def __init__(self, channels, param_dim=2, hidden_dim=32):
        super().__init__()
        self.fc1_weight = nn.Linear(param_dim, hidden_dim)
        self.fc1_bias = nn.Linear(param_dim, hidden_dim)
        self.fc2_weight = nn.Linear(hidden_dim, channels)
        self.fc2_bias = nn.Linear(hidden_dim, channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, params):
        weight = self.fc2_weight(self.act(self.fc1_weight(params)))
        bias = self.fc2_bias(self.act(self.fc1_bias(params)))
        return weight, bias


def apply_param_injection(feat, weight, bias):
    outputs = []
    for idx in range(feat.size(0)):
        feat_n = F.instance_norm(feat.narrow(0, idx, 1), weight=weight[idx], bias=bias[idx])
        outputs.append(feat_n)
    return torch.cat(outputs, dim=0)


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels, dilation, norm=AdaptiveNorm):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False
        )
        self.norm1 = norm(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False
        )
        self.norm2 = norm(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class LowResBaseNet(nn.Module):
    def __init__(self, channels=48):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            AdaptiveNorm(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        dilations = [1, 2, 4, 8, 4, 2]
        self.body = nn.Sequential(
            *[ResidualDilatedBlock(channels, dilation=d, norm=AdaptiveNorm) for d in dilations]
        )
        self.inject_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.param_condition = ParamConditioning(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.out_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x_lr, params):
        feat = self.head(x_lr)
        feat = self.body(feat)
        feat = self.inject_conv(feat)
        norm_weight, norm_bias = self.param_condition(params)
        feat = self.act(apply_param_injection(feat, norm_weight, norm_bias))
        return self.out_conv(feat)


class WeightPredictor(nn.Module):
    def __init__(self, in_channels=4, channels=32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.inject_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.param_condition = ParamConditioning(channels)
        self.act = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(channels, 3, kernel_size=1, bias=True)

    def forward(self, feat_in, params):
        feat = self.head(feat_in)
        feat = self.body(feat)
        feat = self.inject_conv(feat)
        norm_weight, norm_bias = self.param_condition(params)
        feat = self.act(apply_param_injection(feat, norm_weight, norm_bias))
        return self.out_conv(feat)


def compute_anisotropic_weights(guide, alpha_map, eps=1e-4):
    grad_x = torch.abs(guide[:, :, :, 1:] - guide[:, :, :, :-1])
    grad_y = torch.abs(guide[:, :, 1:, :] - guide[:, :, :-1, :])
    w_x = torch.pow(grad_x + eps, -alpha_map)
    w_y = torch.pow(grad_y + eps, -alpha_map)
    w_x = w_x / (1.0 + w_x)
    w_y = w_y / (1.0 + w_y)
    return w_x, w_y


def weighted_laplacian(u, w_x, w_y):
    grad_x = u[:, :, :, 1:] - u[:, :, :, :-1]
    grad_y = u[:, :, 1:, :] - u[:, :, :-1, :]
    flux_x = w_x * grad_x
    flux_y = w_y * grad_y

    lap = torch.zeros_like(u)
    lap[:, :, :, :-1] += flux_x
    lap[:, :, :, 1:] -= flux_x
    lap[:, :, :-1, :] += flux_y
    lap[:, :, 1:, :] -= flux_y
    return lap


class WLSUpdateBlock(nn.Module):
    def __init__(self, channels=32, source_scale=0.05):
        super().__init__()
        self.predictor = WeightPredictor(in_channels=4, channels=channels)
        self.logit_eta = nn.Parameter(torch.tensor(-1.4))
        self.source_scale = source_scale

    def forward(self, guide, base, current, params):
        _, _, guide_mag = _sobel_grad(guide)
        guide_mag = guide_mag / (guide_mag.amax(dim=(2, 3), keepdim=True) + 1e-6)
        feat_in = torch.cat([guide, base, current, guide_mag], dim=1)
        pred = self.predictor(feat_in, params)
        mod_x = pred[:, 0:1]
        mod_y = pred[:, 1:2]
        source = pred[:, 2:3]

        lambda_raw = params[:, 0:1].view(-1, 1, 1, 1)
        alpha_raw = params[:, 1:2].view(-1, 1, 1, 1)

        lambda_eff = torch.log1p(lambda_raw.clamp_min(0.0))
        alpha_eff = alpha_raw.clamp(0.05, 4.0)

        prior_wx, prior_wy = compute_anisotropic_weights(guide, alpha_eff)
        w_x = prior_wx * (0.5 + torch.sigmoid(mod_x[:, :, :, :-1]))
        w_y = prior_wy * (0.5 + torch.sigmoid(mod_y[:, :, :-1, :]))

        lap = weighted_laplacian(current, w_x, w_y)
        eta = torch.sigmoid(self.logit_eta)
        source_term = self.source_scale * torch.tanh(source)
        updated = current - eta * ((current - base) + lambda_eff * lap) + source_term
        return updated, {
            "wx_mean": w_x.mean(),
            "wy_mean": w_y.mean(),
            "eta": eta,
        }


class LearnedWLSNet(nn.Module):
    def __init__(self, num_steps=5, lr_channels=48, step_channels=32):
        super().__init__()
        self.lr_net = LowResBaseNet(channels=lr_channels)
        self.steps = nn.ModuleList([WLSUpdateBlock(channels=step_channels) for _ in range(num_steps)])

    def forward(self, x_lr, x_hr, params):
        base_lr = self.lr_net(x_lr, params)
        base_hr = F.interpolate(base_lr, size=x_hr.shape[-2:], mode="bilinear", align_corners=True)
        current = base_hr
        debug = []
        for step in self.steps:
            current, info = step(x_hr, base_hr, current, params)
            debug.append(info)
        output = current.clamp(0.0, 1.0)
        return {
            "base_lr": base_lr,
            "base_hr": base_hr,
            "final": output,
            "debug": debug,
        }


def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100.0 if mse == 0 else 10.0 * np.log10(1.0 / mse)
    return psnr, mse


def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, save_dir, is_best=False):
    full_path = os.path.join(save_dir, "learned_wls_full.pth")
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": best_loss,
        },
        full_path,
    )
    print(f"Checkpoint saved: {full_path}")

    last_model_path = os.path.join(save_dir, "learned_wls_last.pth")
    torch.save(model.state_dict(), last_model_path)
    print(f"Last model weights saved: {last_model_path}")

    if is_best:
        best_model_path = os.path.join(save_dir, "learned_wls_best.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"*** Best model updated: {best_model_path} (Test MSE: {best_loss:.6f}) ***")


downsample_folder_train = "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/traininput_downsampled"
input_folder_train = "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/traininput"
label_train = "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/trainlabel"

downsample_folder_test = "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/testinput_downsampled"
input_folder_test = "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/testinput"
label_test = "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/testlabel"


def build_dataloaders():
    train_dataset = GuidedFilterDataset(
        x_lr_dir=downsample_folder_train,
        x_hr_dir=input_folder_train,
        label_dir=label_train,
    )
    test_dataset = GuidedFilterDataset(
        x_lr_dir=downsample_folder_test,
        x_hr_dir=input_folder_test,
        label_dir=label_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=var_custom_collate,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        collate_fn=var_custom_collate,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader


def prepare_model(device):
    model = LearnedWLSNet(num_steps=args.steps)
    return torch.nn.DataParallel(model).to(device)


def load_checkpoint_if_needed(model, optimizer, scheduler, device):
    os.makedirs(args.save_dir, exist_ok=True)
    resume_path = args.resume_path or os.path.join(args.save_dir, "learned_wls_full.pth")
    start_epoch = 0
    best_loss = float("inf")

    if args.no_resume:
        print("=> Resume disabled, training from scratch.")
        return start_epoch, best_loss

    if os.path.exists(resume_path):
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

    return start_epoch, best_loss


def main():
    seed_everything(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = build_dataloaders()
    model = prepare_model(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.MSELoss(reduction="mean")
    start_epoch, best_loss = load_checkpoint_if_needed(model, optimizer, scheduler, device)

    task_name = "learned_wls_unrolled"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_root, f"{task_name}_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in tqdm(range(start_epoch, args.epochs), desc="Training Progress", dynamic_ncols=True):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_psnr = 0.0

        for batch_idx, (batch_x_lr, batch_x_hr, batch_label, batch_params) in enumerate(train_loader):
            batch_x_hr = crop_to_even(batch_x_hr)
            batch_label = crop_to_even(batch_label)

            batch_x_lr = batch_x_lr.to(device)
            # Keep the same centering convention as the guided-filter baseline:
            # HR input is treated as the guide and centered around zero.
            batch_x_hr = batch_x_hr.to(device) - 0.5
            batch_label = batch_label.to(device)
            batch_params = batch_params.to(device)

            outputs = model(batch_x_lr, batch_x_hr, batch_params)
            final = outputs["final"]
            base_hr = outputs["base_hr"]

            loss_mse = criterion(final, batch_label)
            loss_grad = sobel_gradient_loss(final, batch_label)
            loss_base = criterion(base_hr, batch_label)
            scale_grad = (loss_grad / (loss_mse + 1e-8)).detach().clamp_min(1e-8)
            loss = loss_mse + loss_grad / scale_grad + 0.2 * loss_base

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr, mse = calc_psnr(final.detach(), batch_label)
            epoch_loss += loss.item()
            epoch_mse += mse
            epoch_psnr += psnr

            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/mse", mse, global_step)
                writer.add_scalar("train/psnr", psnr, global_step)
                if outputs["debug"]:
                    writer.add_scalar("train/wx_mean", outputs["debug"][-1]["wx_mean"].item(), global_step)
                    writer.add_scalar("train/wy_mean", outputs["debug"][-1]["wy_mean"].item(), global_step)
                    writer.add_scalar("train/eta", outputs["debug"][-1]["eta"].item(), global_step)

            if batch_idx % 200 == 0:
                print(f"iter loss: {loss.item():.6f}, mse: {mse:.6f}, psnr: {psnr:.2f}")

        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_mse = epoch_mse / len(train_loader)
        avg_epoch_psnr = epoch_psnr / len(train_loader)
        print(
            f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_epoch_loss:.6f} "
            f"- MSE: {avg_epoch_mse:.6f} - PSNR: {avg_epoch_psnr:.2f} dB"
        )
        writer.add_scalar("train/loss_epoch", avg_epoch_loss, epoch)
        writer.add_scalar("train/mse_epoch", avg_epoch_mse, epoch)
        writer.add_scalar("train/psnr_epoch", avg_epoch_psnr, epoch)

        model.eval()
        test_loss = 0.0
        test_mse = 0.0
        test_psnr = 0.0
        with torch.no_grad():
            for test_x_lr, test_x_hr, test_label, test_params in test_loader:
                test_x_hr = crop_to_even(test_x_hr)
                test_label = crop_to_even(test_label)

                test_x_lr = test_x_lr.to(device)
                test_x_hr = test_x_hr.to(device) - 0.5
                test_label = test_label.to(device)
                test_params = test_params.to(device)

                outputs = model(test_x_lr, test_x_hr, test_params)
                test_output = outputs["final"]
                base_hr = outputs["base_hr"]

                v_loss_mse = criterion(test_output, test_label)
                v_loss_grad = sobel_gradient_loss(test_output, test_label)
                v_loss_base = criterion(base_hr, test_label)
                v_scale_grad = (v_loss_grad / (v_loss_mse + 1e-8)).detach().clamp_min(1e-8)
                v_loss = v_loss_mse + v_loss_grad / v_scale_grad + 0.2 * v_loss_base

                psnr, mse = calc_psnr(test_output, test_label)
                test_loss += v_loss.item()
                test_mse += mse
                test_psnr += psnr

        avg_test_loss = test_loss / len(test_loader)
        avg_test_mse = test_mse / len(test_loader)
        avg_test_psnr = test_psnr / len(test_loader)
        print(
            f"Epoch {epoch + 1}/{args.epochs} - Test Loss: {avg_test_loss:.6f} "
            f"- Test MSE: {avg_test_mse:.6f} - Test PSNR: {avg_test_psnr:.2f} dB"
        )
        writer.add_scalar("test/loss", avg_test_loss, epoch)
        writer.add_scalar("test/mse", avg_test_mse, epoch)
        writer.add_scalar("test/psnr", avg_test_psnr, epoch)

        is_best = avg_test_mse < best_loss
        if is_best:
            best_loss = avg_test_mse

        save_checkpoint(model, optimizer, scheduler, epoch, avg_test_mse, args.save_dir, is_best=is_best)

    writer.close()


if __name__ == "__main__":
    main()
