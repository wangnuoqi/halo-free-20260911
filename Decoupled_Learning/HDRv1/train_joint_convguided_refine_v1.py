import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

import os
import datetime
from tqdm import tqdm
from guided_filter_pytorch.box_filter import BoxFilter
import cv2
import numpy as np
import random


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()
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
        super(ConvGuidedFilter, self).__init__()
        self.box_filter = nn.Conv2d(
            1, 1, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=1
        )
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

        N = self.box_filter(x_lr.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0))
        mean_x = self.box_filter(x_lr) / N
        mean_y = self.box_filter(y_lr) / N
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y
        var_x = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x

        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        b = mean_y - A * mean_x

        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode="bilinear", align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode="bilinear", align_corners=True)

        return mean_A * x_hr + mean_b


class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        N = self.boxfilter(x.new_ones((1, 1, h_x, w_x)))
        mean_x = self.boxfilter(x) / N
        mean_y = self.boxfilter(y) / N
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x

        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b


def build_lr_net(norm=AdaptiveNorm, layer=5):
    layers = [
        nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
    ]

    for l in range(1, layer):
        layers += [
            nn.Conv2d(
                24, 24, kernel_size=3, stride=1, padding=2 ** l, dilation=2 ** l, bias=False
            ),
            norm(24),
            nn.LeakyReLU(0.2, inplace=True),
        ]

    layers += [
        nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(24, 1, kernel_size=1, stride=1, padding=0, dilation=1),
    ]

    net = nn.Sequential(*layers)
    net.apply(weights_init_identity)
    return net


class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8):
        super(DeepGuidedFilter, self).__init__()
        self.lr = build_lr_net()
        self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm)

    def forward(self, x_lr, x_hr):
        return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)


class EdgeAwareRefineNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
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

    def forward(self, x_raw, x_coarse):
        g, _, _ = compute_sobel_grad(x_coarse, normalize=True)
        inp = torch.cat([x_raw, x_coarse, g], dim=1)
        delta = self.net(inp)
        out = x_coarse + delta
        return out


class JointHDRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convguided = DeepGuidedFilter(radius=1, eps=1e-8)
        self.refine = EdgeAwareRefineNet(in_channels=3)

    def forward(self, x_lr, x_hr):
        coarse = self.convguided(x_lr, x_hr)
        refined = self.refine(x_hr, coarse)
        return coarse, refined


def compute_sobel_grad(x, eps=1e-6, normalize=False):
    sobel_x = torch.tensor(
        [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]],
        device=x.device,
        dtype=x.dtype,
    )
    sobel_y = torch.tensor(
        [[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]],
        device=x.device,
        dtype=x.dtype,
    )
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + eps)
    if normalize:
        g = g / (g.amax(dim=(2, 3), keepdim=True) + eps)
    return g, gx, gy


def per_channel_loss(pred, target, loss_type="l1"):
    if loss_type == "l1":
        return (pred - target).abs()
    if loss_type == "l2":
        d = pred - target
        return d * d
    if loss_type == "huber":
        return F.smooth_l1_loss(pred, target, reduction="none")
    raise ValueError("loss_type must be 'l1' | 'l2' | 'huber'")


def masked_reduce(per, mask=None, eps=1e-8, reduction="mean"):
    if mask is not None:
        per = per * mask
        if reduction == "mean":
            denom = mask.sum().clamp_min(eps)
            return per.sum() / denom
        if reduction == "sum":
            return per.sum()
        return per
    if reduction == "mean":
        return per.mean()
    if reduction == "sum":
        return per.sum()
    return per


def edge_guided_loss(
    pred,
    input,
    target,
    thresh_mode="percentile",
    thresh=0.6,
    dilate=1,
    loss_type="l1",
    eps=1e-6,
    normalize_grad=True,
):
    assert pred.shape == target.shape, "pred/target 形状必须一致"
    g, _, _ = compute_sobel_grad(target, eps=eps, normalize=normalize_grad)

    if thresh_mode == "fixed":
        T = torch.as_tensor(thresh, device=pred.device, dtype=pred.dtype)
        mask = (g > T).float()
    elif thresh_mode == "percentile":
        q = torch.quantile(g.flatten(1), torch.as_tensor(thresh, device=g.device), dim=1)
        T = q.view(-1, 1, 1, 1)
        mask = (g > T).float()
    else:
        raise ValueError("thresh_mode must be 'fixed' or 'percentile'")

    if dilate and dilate > 0:
        k = dilate * 2 + 1
        mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=dilate)

    per_pix = per_channel_loss(pred, target, loss_type=loss_type)
    pix_loss = masked_reduce(per_pix, mask, eps=eps, reduction="mean")

    gp, gpx, gpy = compute_sobel_grad(pred, eps=eps, normalize=normalize_grad)
    gt, gtx, gty = compute_sobel_grad(target, eps=eps, normalize=normalize_grad)
    grad_loss = (gpx - gtx).abs() + (gpy - gty).abs()
    grad_loss = grad_loss * mask
    denom = mask.sum().clamp_min(eps)
    grad_loss = grad_loss.sum() / denom

    eps = 1e-8
    total = pix_loss + grad_loss / ((grad_loss / (pix_loss + eps)).detach())
    return total


def _to_gray(x):
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.size(1) == 1:
        return x
    return x.mean(dim=1, keepdim=True)


def _sobel_grad(x, eps=1e-6):
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


def crop_to_even(tensor):
    _, _, H, W = tensor.shape
    new_H = H if H % 2 == 0 else H - 1
    new_W = W if W % 2 == 0 else W - 1
    return tensor[:, :, :new_H, :new_W]


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
        input_hr_h, input_hr_w = x_hr.shape
        input_lr_h, input_lr_w = x_lr.shape

        return {
            "input_lr": torch.from_numpy(x_lr).unsqueeze(0).unsqueeze(0),
            "input_hr": torch.from_numpy(x_hr).unsqueeze(0).unsqueeze(0),
            "target_img": torch.from_numpy(label).unsqueeze(0).unsqueeze(0),
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

    for idx, item in enumerate(batch):
        x_lr_crop = item["input_lr"][:, :, :min_h_lr, :min_w_lr]
        x_hr_crop = item["input_hr"][:, :, :min_h_hr, :min_w_hr]
        label_crop = item["target_img"][:, :, :min_h_hr, :min_w_hr]

        batch_x_lr[idx] = x_lr_crop
        batch_x_hr[idx] = x_hr_crop
        batch_target_images[idx] = label_crop

    return batch_x_lr, batch_x_hr, batch_target_images


def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1 / mse)
    return psnr, mse


downsample_folder_train = "/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/down_sample_2_traininput"
input_folder_train = "/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/traininput"
label_train = "/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/trainlabel"

downsample_folder_test = "/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/down_sample_2_testinput"
input_folder_test = "/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/testinput"
label_test = "/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/testlabel"


train_dataset = GuidedFilterDataset(
    x_lr_dir=downsample_folder_train, x_hr_dir=input_folder_train, label_dir=label_train
)
test_dataset = GuidedFilterDataset(
    x_lr_dir=downsample_folder_test, x_hr_dir=input_folder_test, label_dir=label_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    collate_fn=var_custom_collate,
    pin_memory=True,
    drop_last=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=var_custom_collate,
    pin_memory=True,
    drop_last=False,
)


device = torch.device("cuda:0")
model = JointHDRNet()
model = torch.nn.DataParallel(model).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
best_loss = float("inf")
save_dir = "/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/models_retrain"
epochs = 80


task_name = "joint_convguided_refine_v1"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(
    "/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/log_dir", f"{task_name}_{timestamp}"
)
writer = SummaryWriter(log_dir=log_dir)


def save_checkpoint(epoch, best_loss, is_best=False):
    full_path = os.path.join(save_dir, "joint_convguided_refine_full_v1.pth")
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

    last_model_path = os.path.join(save_dir, "joint_convguided_refine_last_v1.pth")
    torch.save(model.state_dict(), last_model_path)
    print(f"Last model weights saved: {last_model_path}")

    if is_best:
        best_model_path = os.path.join(save_dir, "joint_convguided_refine_best_v1.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"*** Best model updated: {best_model_path} (Test MSE: {best_loss:.6f}) ***")


resume_path = os.path.join(save_dir, "joint_convguided_refine_full_v1.pth")
start_epoch = 0
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


if __name__ == "__main__":
    criterion = nn.MSELoss(reduction="mean")

    for epoch in tqdm(range(start_epoch, epochs), desc="Training Progress", dynamic_ncols=True):
        model.train()
        epoch_mse = 0
        epoch_psnr = 0
        for batch_idx, (batch_x_lr, batch_x_hr, batch_label) in enumerate(train_loader):
            batch_x_hr = crop_to_even(batch_x_hr)
            batch_label = crop_to_even(batch_label)

            batch_x_lr = batch_x_lr.to(device)
            batch_x_hr = batch_x_hr.to(device)
            batch_label = batch_label.to(device)

            batch_x_lr = batch_x_lr - 0.5
            batch_x_hr = batch_x_hr - 0.5

            coarse, refined = model(batch_x_lr, batch_x_hr)

            loss_mse = criterion(coarse, batch_label)
            loss_g = sobel_gradient_loss(coarse, batch_label, mode="vector")
            scale_g = (loss_g / (loss_mse + 1e-8)).detach().clamp_min(1e-8)
            conv_loss = loss_mse + loss_g / scale_g

            batch_label_shift = batch_label - 0.5
            coarse_shift = coarse - 0.5
            refine_loss = edge_guided_loss(
                refined,
                coarse_shift,
                batch_label_shift,
                thresh_mode="percentile",
                thresh=0.6,
                dilate=0,
                loss_type="l1",
                normalize_grad=True,
            )

            loss = conv_loss + refine_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            refined_vis = torch.clamp(refined + 0.5, 0, 1)
            psnr, mse = calc_psnr(refined_vis.detach(), batch_label)
            if batch_idx % 200 == 0:
                print(f"iter MSE loss: {mse:.6f}")
            epoch_mse += mse
            epoch_psnr += psnr

            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 50 == 0:
                writer.add_scalar("train/mse", mse, global_step)
                writer.add_scalar("train/psnr", psnr, global_step)
                writer.add_scalar("train/conv_loss", conv_loss.item(), global_step)
                writer.add_scalar("train/refine_loss", refine_loss.item(), global_step)

        avg_epoch_mse = epoch_mse / len(train_loader)
        avg_epoch_psnr = epoch_psnr / len(train_loader)
        print(
            f"Epoch {epoch+1}/{epochs} "
            f"- MSE: {avg_epoch_mse:.6f} "
            f"- PSNR: {avg_epoch_psnr:.2f} dB "
        )
        writer.add_scalar("train/mse_epoch", avg_epoch_mse, epoch)
        writer.add_scalar("train/psnr_epoch", avg_epoch_psnr, epoch)
        scheduler.step()

        model.eval()
        test_mse = 0
        test_psnr = 0
        with torch.no_grad():
            for test_x_lr, test_x_hr, test_label in test_loader:
                test_x_hr = crop_to_even(test_x_hr)
                test_label = crop_to_even(test_label)

                test_x_lr = test_x_lr.to(device)
                test_x_hr = test_x_hr.to(device)
                test_label = test_label.to(device)

                test_x_lr = test_x_lr - 0.5
                test_x_hr = test_x_hr - 0.5

                coarse, refined = model(test_x_lr, test_x_hr)
                refined_vis = torch.clamp(refined + 0.5, 0, 1)

                psnr, mse = calc_psnr(refined_vis, test_label)
                test_mse += mse
                test_psnr += psnr

        avg_test_mse = test_mse / len(test_loader)
        avg_test_psnr = test_psnr / len(test_loader)

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"- Test MSE: {avg_test_mse:.6f} "
            f"- Test PSNR: {avg_test_psnr:.2f} dB"
        )
        writer.add_scalar("test/mse", avg_test_mse, epoch)
        writer.add_scalar("test/psnr", avg_test_psnr, epoch)

        is_best = avg_test_mse < best_loss
        if is_best:
            best_loss = avg_test_mse
        save_checkpoint(epoch, best_loss, is_best=is_best)

    writer.close()
