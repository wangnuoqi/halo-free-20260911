import os
import datetime
import random
import math

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from guided_filter_pytorch.box_filter import BoxFilter


# ===================== 工具函数 & 随机种子 =====================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


# ===================== 模型相关 =====================

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
    if classname.find('Conv') != -1:
        n_out, n_in, h, w = m.weight.data.size()
        if n_out < n_in:
            nn.init.xavier_uniform_(m.weight.data)
            return
        m.weight.data.zero_()
        ch, cw = h // 2, w // 2
        for i in range(n_in):
            m.weight.data[i, i, ch, cw] = 1.0
    elif classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)


class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1, norm=nn.BatchNorm2d):
        super(ConvGuidedFilter, self).__init__()
        print("Loaded ConvGuidedFilter from:", __file__)
        self.box_filter = nn.Conv2d(
            1, 1, kernel_size=3, padding=radius, dilation=radius,
            bias=False, groups=1
        )
        self.conv_a = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=False)
        )
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0))
        # mean_x
        mean_x = self.box_filter(x_lr) / N
        # mean_y
        mean_y = self.box_filter(y_lr) / N
        # cov_xy
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y
        # var_x
        var_x = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x

        # A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

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

        # N
        N = self.boxfilter(x.new_ones((1, 1, h_x, w_x)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
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
                24, 24,
                kernel_size=3,
                stride=1,
                padding=2 ** l,
                dilation=2 ** l,
                bias=False
            ),
            norm(24),
            nn.LeakyReLU(0.2, inplace=True)
        ]

    layers += [
        nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(24, 1, kernel_size=1, stride=1, padding=0, dilation=1)
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

    def init_lr(self, path):
        self.lr.load_state_dict(torch.load(path))


# ===================== 数据路径 =====================

downsample_folder_train = "D:/nuoqi/S2R-HDR/data/lavelindoor_hdr/down_sample_2_train_crop"
input_folder_train = 'D:/nuoqi/S2R-HDR/data/lavelindoor_hdr/traininput_crop'
label_train = "D:/nuoqi/S2R-HDR/data/lavelindoor_hdr/trainlabel_crop"

downsample_folder_test = 'D:/nuoqi/S2R-HDR/data/lavelindoor_hdr/down_sample_2_test_crop'
input_folder_test = 'D:/nuoqi/S2R-HDR/data/lavelindoor_hdr/testinput_crop'
label_test = 'D:/nuoqi/S2R-HDR/data/lavelindoor_hdr/testlabel_crop'


# ===================== Dataset & collate =====================

class GuidedFilterDataset(Dataset):
    def __init__(self, x_lr_dir, x_hr_dir, label_dir):
        self.x_lr_files = sorted([os.path.join(x_lr_dir, f) for f in os.listdir(x_lr_dir)])
        self.x_hr_files = sorted([os.path.join(x_hr_dir, f) for f in os.listdir(x_hr_dir)])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        print(f"LR: {len(self.x_lr_files)}, HR: {len(self.x_hr_files)}, Label: {len(self.label_files)}")

    def __len__(self):
        return min(len(self.x_lr_files), len(self.x_hr_files), len(self.label_files))

    def __getitem__(self, idx):
        x_lr = cv2.imread(self.x_lr_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        x_hr = cv2.imread(self.x_hr_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        label = cv2.imread(self.label_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        input_hr_h, input_hr_w = x_hr.shape
        input_lr_h, input_lr_w = x_lr.shape

        return {
            'input_lr': torch.from_numpy(x_lr).unsqueeze(0).unsqueeze(0),  # [1,1,H,W]
            'input_hr': torch.from_numpy(x_hr).unsqueeze(0).unsqueeze(0),
            'target_img': torch.from_numpy(label).unsqueeze(0).unsqueeze(0),
            'input_height_hr': input_hr_h,
            'input_width_hr': input_hr_w,
            'input_height_lr': input_lr_h,
            'input_width_lr': input_lr_w
        }


def var_custom_collate(batch):
    min_h_hr = min([item['input_height_hr'] for item in batch])
    min_w_hr = min([item['input_width_hr'] for item in batch])
    min_h_lr = min([item['input_height_lr'] for item in batch])
    min_w_lr = min([item['input_width_lr'] for item in batch])

    batch_x_hr = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_x_lr = torch.zeros(len(batch), 1, min_h_lr, min_w_lr)
    batch_target_images = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)

    for idx, item in enumerate(batch):
        x_lr_crop = item['input_lr'][:, :, :min_h_lr, :min_w_lr]
        x_hr_crop = item['input_hr'][:, :, :min_h_hr, :min_w_hr]
        label_crop = item['target_img'][:, :, :min_h_hr, :min_w_hr]

        batch_x_lr[idx] = x_lr_crop
        batch_x_hr[idx] = x_hr_crop
        batch_target_images[idx] = label_crop

    return batch_x_lr, batch_x_hr, batch_target_images


# ===================== 其他工具函数 =====================

def crop_to_even(tensor):
    """
    将输入 Tensor 裁剪到高宽为偶数，保持左上角对齐（裁掉底部和右边）。
    tensor: [B, C, H, W]
    """
    _, _, H, W = tensor.shape
    new_H = H if H % 2 == 0 else H - 1
    new_W = W if W % 2 == 0 else W - 1
    return tensor[:, :, :new_H, :new_W]


def _to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:  # (B,H,W)
        x = x.unsqueeze(1)
    if x.size(1) == 1:
        return x
    return x.mean(dim=1, keepdim=True)


def _sobel_grad(x: torch.Tensor, eps: float = 1e-6):
    x = F.pad(x, (1, 1, 1, 1), mode='reflect')
    kx = torch.tensor(
        [[[[-1., 0., 1.],
           [-2., 0., 2.],
           [-1., 0., 1.]]]],
        device=x.device, dtype=x.dtype
    )
    ky = torch.tensor(
        [[[[-1., -2., -1.],
           [0., 0., 0.],
           [1., 2., 1.]]]],
        device=x.device, dtype=x.dtype
    )
    gx = F.conv2d(x, kx)
    gy = F.conv2d(x, ky)
    mag = torch.sqrt(gx * gx + gy * gy + eps)
    return gx, gy, mag


def sobel_gradient_loss(pred: torch.Tensor,
                        target: torch.Tensor,
                        mode: str = 'vector',
                        reduction: str = 'mean',
                        eps: float = 1e-6) -> torch.Tensor:
    pred_g = _to_gray(pred)
    targ_g = _to_gray(target)

    pgx, pgy, pm = _sobel_grad(pred_g, eps)
    tgx, tgy, tm = _sobel_grad(targ_g, eps)

    if mode == 'vector':
        loss = (pgx - tgx).abs() + (pgy - tgy).abs()
    elif mode == 'magnitude':
        loss = (pm - tm).abs()
    else:
        raise ValueError("mode must be 'vector' or 'magnitude'")

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError("reduction must be 'mean' | 'sum' | 'none'")


def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1 / mse)
    return psnr, mse


def save_checkpoint(epoch, best_loss, model, optimizer, scheduler, save_dir, is_best=False):
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(model, (DDP, nn.DataParallel)):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    full_path = os.path.join(save_dir, "convguided_filter_full_lavel_crop.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss
    }, full_path)
    print(f"[Rank 0] Checkpoint saved: {full_path}")

    last_model_path = os.path.join(save_dir, "convguided_filter_last_lavel_crop.pth")
    torch.save(model_state, last_model_path)
    print(f"[Rank 0] Last model weights saved: {last_model_path}")

    if is_best:
        best_model_path = os.path.join(save_dir, f"convguided_filter_best_lavel_crop.pth")
        torch.save(model_state, best_model_path)
        print(f"[Rank 0] *** Best model updated: {best_model_path} (Test MSE: {best_loss:.6f}) ***")


# ===================== DDP 单机多卡 worker =====================

def train_worker(rank, world_size):
    print(f"[Rank {rank}] Starting, world_size={world_size}")

    # 明确指定 tcp 初始化，避免 env:// 需要 RANK/WORLD_SIZE 环境变量
    dist.init_process_group(
        backend="gloo",  # Windows 用 gloo
        init_method="tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    seed_everything(42 + rank)

    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    model = DeepGuidedFilter(radius=1, eps=1e-8).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Dataset & Sampler
    train_dataset = GuidedFilterDataset(
        x_lr_dir=downsample_folder_train,
        x_hr_dir=input_folder_train,
        label_dir=label_train
    )
    test_dataset = GuidedFilterDataset(
        x_lr_dir=downsample_folder_test,
        x_hr_dir=input_folder_test,
        label_dir=label_test
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    per_gpu_batch_size = 1  # 每张卡一张图，显存够可以调大
    num_workers = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=var_custom_collate,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=per_gpu_batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        collate_fn=var_custom_collate,
        pin_memory=True,
        drop_last=False
    )

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    task_name = "convguided_filter_lavel"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_root = "D:/nuoqi/halo-free/Decoupled_Learning/HDRv1/log_dir"
    save_dir = "D:/nuoqi/halo-free/Decoupled_Learning/HDRv1/models_retrain"
    os.makedirs(save_dir, exist_ok=True)

    if rank == 0:
        log_dir = os.path.join(log_dir_root, f"{task_name}_{timestamp}")
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[Rank 0] TensorBoard log dir: {log_dir}")
    else:
        writer = None

    best_loss = float("inf")
    epochs = 80
    eps = 1e-8

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)

        model.train()
        epoch_mse = 0.0
        epoch_psnr = 0.0

        if rank == 0:
            loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", dynamic_ncols=True)
        else:
            loader_iter = train_loader

        for batch_idx, (batch_x_lr, batch_x_hr, batch_label) in enumerate(loader_iter):
            batch_x_hr = crop_to_even(batch_x_hr)
            batch_label = crop_to_even(batch_label)

            batch_x_lr = batch_x_lr.to(device, non_blocking=True)
            batch_x_hr = batch_x_hr.to(device, non_blocking=True)
            batch_label = batch_label.to(device, non_blocking=True)

            batch_x_lr = batch_x_lr - 0.5
            batch_x_hr = batch_x_hr - 0.5

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                output = model(batch_x_lr, batch_x_hr)
                loss_mse = criterion(output, batch_label)
                loss_g = sobel_gradient_loss(output, batch_label, mode='vector')
                scale_g = (loss_g / (loss_mse + eps)).detach().clamp_min(eps)
                loss = loss_mse + loss_g / scale_g

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                psnr, mse = calc_psnr(output.detach().float(), batch_label.float())

            epoch_mse += mse
            epoch_psnr += psnr

            if rank == 0:
                if batch_idx % 200 == 0:
                    print(f"[Epoch {epoch+1}][Iter {batch_idx}] MSE: {mse:.6f}")
                global_step = epoch * len(train_loader) + batch_idx
                if batch_idx % 50 == 0 and writer is not None:
                    writer.add_scalar('train/mse', mse, global_step)
                    writer.add_scalar('train/psnr', psnr, global_step)

        # 聚合 train 指标
        epoch_mse_tensor = torch.tensor(epoch_mse, device=device)
        epoch_psnr_tensor = torch.tensor(epoch_psnr, device=device)
        batch_count_tensor = torch.tensor(len(train_loader), device=device, dtype=torch.float32)

        dist.all_reduce(epoch_mse_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_psnr_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_count_tensor, op=dist.ReduceOp.SUM)

        avg_epoch_mse = (epoch_mse_tensor / batch_count_tensor).item()
        avg_epoch_psnr = (epoch_psnr_tensor / batch_count_tensor).item()

        if rank == 0:
            print(f"[Epoch {epoch+1}/{epochs}] "
                  f"Train MSE: {avg_epoch_mse:.6f}  "
                  f"Train PSNR: {avg_epoch_psnr:.2f} dB")
            if writer is not None:
                writer.add_scalar('train/mse_epoch', avg_epoch_mse, epoch)
                writer.add_scalar('train/psnr_epoch', avg_epoch_psnr, epoch)

        scheduler.step()

        # =========== 验证 ===========
        model.eval()
        test_mse = 0.0
        test_psnr = 0.0

        with torch.no_grad():
            for test_x_lr, test_x_hr, test_label in test_loader:
                test_x_hr = crop_to_even(test_x_hr)
                test_label = crop_to_even(test_label)

                test_x_lr = test_x_lr.to(device, non_blocking=True)
                test_x_hr = test_x_hr.to(device, non_blocking=True)
                test_label = test_label.to(device, non_blocking=True)

                test_x_lr = test_x_lr - 0.5
                test_x_hr = test_x_hr - 0.5

                with autocast(enabled=use_amp):
                    test_output = model(test_x_lr, test_x_hr)
                    v_loss = criterion(test_output, test_label)

                psnr, mse = calc_psnr(test_output.float(), test_label.float())
                test_mse += mse
                test_psnr += psnr

        test_mse_tensor = torch.tensor(test_mse, device=device)
        test_psnr_tensor = torch.tensor(test_psnr, device=device)
        test_batch_count_tensor = torch.tensor(len(test_loader), device=device, dtype=torch.float32)

        dist.all_reduce(test_mse_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_psnr_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_batch_count_tensor, op=dist.ReduceOp.SUM)

        avg_test_mse = (test_mse_tensor / test_batch_count_tensor).item()
        avg_test_psnr = (test_psnr_tensor / test_batch_count_tensor).item()

        if rank == 0:
            print(f"[Epoch {epoch+1}/{epochs}] "
                  f"Test MSE: {avg_test_mse:.6f}  "
                  f"Test PSNR: {avg_test_psnr:.2f} dB")
            if writer is not None:
                writer.add_scalar('test/mse', avg_test_mse, epoch)
                writer.add_scalar('test/psnr', avg_test_psnr, epoch)

            is_best = avg_test_mse < best_loss
            if is_best:
                best_loss = avg_test_mse
            save_checkpoint(epoch, avg_test_mse, model, optimizer, scheduler, save_dir, is_best=is_best)

    if rank == 0 and writer is not None:
        writer.close()

    dist.destroy_process_group()
    print(f"[Rank {rank}] Finished.")


if __name__ == "__main__":
    # 用“可见”的所有 GPU；想只用 3 张卡就用 CUDA_VISIBLE_DEVICES 控制
    world_size = torch.cuda.device_count()
    assert world_size >= 1, "No CUDA device available."

    # Windows 默认 spawn
    torch.multiprocessing.spawn(
        train_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
