import os
import time
import math
import random
import datetime

import cv2
import imageio
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from guided_filter_pytorch.box_filter import BoxFilter


# ===================== 工具：固定随机种子 =====================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


# ===================== 模型结构（和训练完全一致） =====================

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
            1, 1,
            kernel_size=3,
            padding=radius,
            dilation=radius,
            bias=False,
            groups=1
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


# ===================== Dataset & collate（和训练版类似，但多返回 filename） =====================

class GuidedFilterEvalDataset(Dataset):
    def __init__(self, x_lr_dir, x_hr_dir, label_dir):
        self.x_lr_files = sorted([os.path.join(x_lr_dir, f) for f in os.listdir(x_lr_dir)])
        self.x_hr_files = sorted([os.path.join(x_hr_dir, f) for f in os.listdir(x_hr_dir)])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        print(f"[Eval] LR: {len(self.x_lr_files)}, HR: {len(self.x_hr_files)}, Label: {len(self.label_files)}")

    def __len__(self):
        return min(len(self.x_lr_files), len(self.x_hr_files), len(self.label_files))

    def __getitem__(self, idx):
        x_lr_path = self.x_lr_files[idx]
        x_hr_path = self.x_hr_files[idx]
        label_path = self.label_files[idx]

        x_lr = cv2.imread(x_lr_path, cv2.IMREAD_UNCHANGED).astype('float32')
        x_hr = cv2.imread(x_hr_path, cv2.IMREAD_UNCHANGED).astype('float32')
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED).astype('float32')

        input_hr_h, input_hr_w = x_hr.shape
        input_lr_h, input_lr_w = x_lr.shape

        filename = os.path.splitext(os.path.basename(x_hr_path))[0]

        return {
            'input_lr': torch.from_numpy(x_lr).unsqueeze(0).unsqueeze(0),  # [1,1,H,W]
            'input_hr': torch.from_numpy(x_hr).unsqueeze(0).unsqueeze(0),
            'target_img': torch.from_numpy(label).unsqueeze(0).unsqueeze(0),
            'filename': filename,
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

    filenames = []

    for idx, item in enumerate(batch):
        x_lr_crop = item['input_lr'][:, :, :min_h_lr, :min_w_lr]
        x_hr_crop = item['input_hr'][:, :, :min_h_hr, :min_w_hr]
        label_crop = item['target_img'][:, :, :min_h_hr, :min_w_hr]

        batch_x_lr[idx] = x_lr_crop
        batch_x_hr[idx] = x_hr_crop
        batch_target_images[idx] = label_crop

        filenames.append(item['filename'])

    return batch_x_lr, batch_x_hr, batch_target_images, filenames


# ===================== 其他工具函数 =====================

def crop_to_even(tensor: torch.Tensor) -> torch.Tensor:
    """
    将输入 Tensor 裁剪到高宽为偶数，保持左上角对齐（裁掉底部和右边）。
    tensor: [B, C, H, W]
    """
    _, _, H, W = tensor.shape
    new_H = H if H % 2 == 0 else H - 1
    new_W = W if W % 2 == 0 else W - 1
    return tensor[:, :, :new_H, :new_W]


def calc_psnr(pred: torch.Tensor, target: torch.Tensor):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1.0 / mse)
    return psnr, mse


def save_float32_tiff(img: torch.Tensor, save_path: str):
    """
    直接保存原始 img 为 float32 tiff，不做归一化，保留动态范围。
    支持 [B,C,H,W] / [C,H,W] / [H,W] 格式。
    """
    arr = img.detach().cpu().numpy()
    if arr.ndim == 4:  # [B,C,H,W]
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 1:  # [1,H,W]
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 3:  # [3,H,W] -> [H,W,3]
        arr = np.transpose(arr, (1, 2, 0))

    arr = arr.astype(np.float32)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.imwrite(save_path, arr)


# ===================== 主评估流程 =====================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # ---------- 路径配置：按你的实际情况改 ----------
    downsample_folder_test = '/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/down_sample_2_test_log'
    input_folder_test = '/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/input_test_log'
    label_test = '/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/testlabel_log'

    # 训练时 save_dir 里的 best 权重
    model_path = "/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/models_retrain/convguided_filter_best_lavel.pth"

    # 输出结果目录
    output_dir = "/mnt/d/nuoqi/S2R-HDR/data/lavelindoor_hdr/result/df_lavelfull"
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 加载模型权重 ----------
    base_model = DeepGuidedFilter(radius=1, eps=1e-8)
    state = torch.load(model_path, map_location=device)

    # 兼容两种保存方式：纯 state_dict 或 含 'model_state_dict'
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    base_model.load_state_dict(state)
    base_model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    model.eval()

    # ---------- 构建 DataLoader ----------
    test_dataset = GuidedFilterEvalDataset(
        x_lr_dir=downsample_folder_test,
        x_hr_dir=input_folder_test,
        label_dir=label_test
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,          # 一张一张评估，方便保存
        shuffle=False,
        num_workers=0,
        collate_fn=var_custom_collate,
        pin_memory=True,
        drop_last=False
    )

    criterion = nn.MSELoss(reduction='mean')

    # ---------- 开始测试 ----------
    total_psnr = 0.0
    total_mse = 0.0
    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for batch_x_lr, batch_x_hr, batch_label, filenames in tqdm(test_loader, desc="Testing"):
            batch_x_hr = crop_to_even(batch_x_hr)
            batch_label = crop_to_even(batch_label)

            batch_x_lr = batch_x_lr.to(device, non_blocking=True)
            batch_x_hr = batch_x_hr.to(device, non_blocking=True)
            batch_label = batch_label.to(device, non_blocking=True)

            # 和训练一致的预处理
            batch_x_lr = batch_x_lr - 0.5
            batch_x_hr = batch_x_hr - 0.5

            start_time = time.time()
            output = model(batch_x_lr, batch_x_hr)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()

            elapsed = end_time - start_time
            total_time += elapsed
            B = output.size(0)
            total_images += B

            v_loss = criterion(output, batch_label)

            psnr, mse = calc_psnr(output, batch_label)
            total_psnr += psnr * B
            total_mse += mse * B

            # 保存结果
            for i, fname in enumerate(filenames):
                save_path = os.path.join(output_dir, f"{fname}_pred.tiff")
                #save_float32_tiff(output[i], save_path)

    avg_psnr = total_psnr / total_images
    avg_mse = total_mse / total_images
    avg_time_per_image = total_time / total_images

    print(f"Test Finished - Average PSNR: {avg_psnr:.2f} dB, Average MSE: {avg_mse:.6f}")
    print(f"Average Inference Time per Image: {avg_time_per_image * 1000:.2f} ms")


if __name__ == "__main__":
    main()
