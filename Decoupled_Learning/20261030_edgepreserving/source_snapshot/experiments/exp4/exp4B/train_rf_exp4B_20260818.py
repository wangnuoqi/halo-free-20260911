import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import torch.nn.functional as F


import os, datetime
from tqdm import tqdm
import cv2
import numpy as np
import time
import math
import imageio

class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))
        self.bn = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


def weights_init_identity(module):
    name = module.__class__.__name__
    if "Conv" in name:
        n_out, n_in, height, width = module.weight.data.size()
        if n_out < n_in:
            nn.init.xavier_uniform_(module.weight.data)
            return
        module.weight.data.zero_()
        center_h, center_w = height // 2, width // 2
        for index in range(n_in):
            module.weight.data[index, index, center_h, center_w] = 1.0
    elif "BatchNorm2d" in name:
        nn.init.constant_(module.weight.data, 1.0)
        nn.init.constant_(module.bias.data, 0.0)


class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1, norm=AdaptiveNorm):
        super().__init__()
        self.box_filter = nn.Conv2d(
            1, 1, kernel_size=3, padding=radius, dilation=radius,
            bias=False, groups=1,
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
        _, _, h_lr, w_lr = x_lr.size()
        _, _, h_hr, w_hr = x_hr.size()
        count = self.box_filter(x_lr.new_ones((1, 1, h_lr, w_lr)))
        mean_x = self.box_filter(x_lr) / count
        mean_y = self.box_filter(y_lr) / count
        cov_xy = self.box_filter(x_lr * y_lr) / count - mean_x * mean_y
        var_x = self.box_filter(x_lr * x_lr) / count - mean_x * mean_x
        a = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        b = mean_y - a * mean_x
        mean_a = F.interpolate(a, (h_hr, w_hr), mode="bilinear", align_corners=True)
        mean_b = F.interpolate(b, (h_hr, w_hr), mode="bilinear", align_corners=True)
        return mean_a * x_hr + mean_b


def build_lr_net(norm=AdaptiveNorm, layer=5):
    layers = [
        nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    for level in range(1, layer):
        layers.extend([
            nn.Conv2d(24, 24, kernel_size=3, padding=2 ** level,
                      dilation=2 ** level, bias=False),
            norm(24),
            nn.LeakyReLU(0.2, inplace=True),
        ])
    layers.extend([
        nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(24, 1, kernel_size=1),
    ])
    network = nn.Sequential(*layers)
    network.apply(weights_init_identity)
    return network


class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8):
        super().__init__()
        self.lr = build_lr_net()
        self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm)

    def forward(self, x_lr, x_hr):
        return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)


def load_frozen_stage1(model_path, device):
    stage1_model = DeepGuidedFilter(radius=1, eps=1e-8).to(device)
    state_dict = torch.load(model_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    stage1_model.load_state_dict(state_dict, strict=True)
    stage1_model.eval()
    for parameter in stage1_model.parameters():
        parameter.requires_grad_(False)
    return stage1_model

class Stage1DeltaNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, 1, kernel_size=1, bias=False)
        )

    def forward(self, x_raw, x_coarse):
        g, _, _ = compute_sobel_grad(x_coarse, normalize=True)
        inp = torch.cat([x_raw, x_coarse, g], dim=1)
        delta = self.net(inp)
        return delta

class ABAdjustNet(nn.Module):
    def __init__(self, base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True):
        super().__init__()
        self.a_scale = a_scale
        self.b_scale = b_scale
        self.use_delta_as_input = use_delta_as_input

        in_ch = 3 + (1 if use_delta_as_input else 0)  # raw + coarse_k + grad + (delta_prev可选)

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, 2, 1, bias=True)  # (a_raw, b_raw) each pixel
        )

    def forward(self, x_raw, x_coarse_k, delta_prev=None):
        """
        x_coarse_k: 每一轮“更新后的 coarse”，来自上一轮 coarse + refine 的和（不是 concat）
        a,b: per-pixel (B,1,H,W)
        """
        g, _, _ = compute_sobel_grad(x_coarse_k, normalize=True)

        if self.use_delta_as_input:
            assert delta_prev is not None
            inp = torch.cat([x_raw, x_coarse_k, g, delta_prev], dim=1)
        else:
            inp = torch.cat([x_raw, x_coarse_k, g], dim=1)

        ab = self.net(inp)
        a_raw = ab[:, 0:1]
        b_raw = ab[:, 1:2]

        # per-pixel a,b（关键：这里就是每个像素一个 a、一个 b）
        a = 1.0 + self.a_scale * torch.tanh(a_raw)
        b = self.b_scale * torch.tanh(b_raw)
        return a, b

class CascadeRefine3Stage(nn.Module):
    def __init__(self, base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True):
        super().__init__()
        self.stage1 = Stage1DeltaNet(base_channels=base_channels)
        self.stage2 = ABAdjustNet(base_channels=base_channels, a_scale=a_scale, b_scale=b_scale,
                                 use_delta_as_input=use_delta_as_input)
        self.stage3 = ABAdjustNet(base_channels=base_channels, a_scale=a_scale, b_scale=b_scale,
                                 use_delta_as_input=use_delta_as_input)

    def forward(self, x_raw, x_coarse0):
        # ---- Stage 1 ----
        delta1 = self.stage1(x_raw, x_coarse0)            # refine delta
        x_coarse1 = x_coarse0 + delta1                    # 更新 coarse（相加）
        # 这里 x_coarse1 就是你说的 “refine output + coarse input 的和”

        # ---- Stage 2 ----
        a2, b2 = self.stage2(x_raw, x_coarse1, delta1)    # per-pixel a,b
        delta2 = a2 * delta1 + b2                         # y = a*x + b
        x_coarse2 = x_coarse0 + delta2                    # 更新 coarse（相加得到下一轮 coarse）

        # ---- Stage 3 ----
        a3, b3 = self.stage3(x_raw, x_coarse2, delta2)
        delta3 = a3 * delta2 + b3
        x_final = x_coarse0 + delta3                      # 最终输出图

        return {
            "delta1": delta1, "coarse1": x_coarse1,
            "delta2": delta2, "coarse2": x_coarse2,
            "delta3": delta3, "final":  x_final,
            "a2": a2, "b2": b2,
            "a3": a3, "b3": b3,
        }

def cascade_loss(outputs, x_coarse0, gt,
                 thresh_mode='otsu', thresh=0.6, dilate=0,
                 loss_type='l1', normalize_grad=True):
    delta_gt = gt - x_coarse0

    l1 = edge_guided_loss(outputs["delta1"],  delta_gt,
                          thresh_mode=thresh_mode, thresh=thresh, dilate=dilate,
                          loss_type=loss_type, normalize_grad=normalize_grad)
    l2 = edge_guided_loss(outputs["delta2"], delta_gt,
                          thresh_mode=thresh_mode, thresh=thresh, dilate=dilate,
                          loss_type=loss_type, normalize_grad=normalize_grad)
    l3 = edge_guided_loss(outputs["delta3"], delta_gt,
                          thresh_mode=thresh_mode, thresh=thresh, dilate=dilate,
                          loss_type=loss_type, normalize_grad=normalize_grad)

    return 0.1*l1 + 0.3*l2 + 0.6*l3, (l1, l2, l3)

# --------- 工具：Sobel 梯度幅值（对灰度图）---------
def compute_sobel_grad(x, eps=1e-6, normalize=False):
    # x: Bx1xHxW (若不是，会自动做灰度)
    #x = to_gray(x)
    sobel_x = torch.tensor([[[[-1., 0., 1.],
                              [-2., 0., 2.],
                              [-1., 0., 1.]]]], device=x.device, dtype=x.dtype)
    sobel_y = torch.tensor([[[[-1., -2., -1.],
                              [ 0.,  0.,  0.],
                              [ 1.,  2.,  1.]]]], device=x.device, dtype=x.dtype)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    g  = torch.sqrt(gx*gx + gy*gy + eps)
    if normalize:
        g = g / (g.amax(dim=(2,3), keepdim=True) + eps)
    return g, gx, gy  # Bx1xHxW

def batched_otsu_threshold(g, num_bins=256, eps=1e-6, min_delta=1e-6):
    """
    g: (B,1,H,W) 梯度幅值图
    返回: (B,) 每张图的 Otsu 阈值（与 g 同物理量纲）
    """
    assert g.dim() == 4 and g.shape[1] == 1, "g must be (B,1,H,W)"
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")

    B = g.shape[0]
    flat = g.flatten(1)

    g_min = flat.min(dim=1).values
    g_max = flat.max(dim=1).values
    g_span = g_max - g_min
    span = g_span.clamp_min(eps)

    norm = ((flat - g_min[:, None]) / span[:, None]).clamp(0.0, 1.0)
    bin_idx = (norm * (num_bins - 1)).long()

    offsets = (torch.arange(B, device=g.device) * num_bins)[:, None]
    hist_idx = (bin_idx + offsets).reshape(-1)
    hist = torch.bincount(hist_idx, minlength=B * num_bins).reshape(B, num_bins).to(g.dtype)

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
    t_norm = centers[0, best_bin]
    thresh = g_min + t_norm * span

    # 极小值保护：避免阈值贴到最大值导致无边缘
    valid_span = g_span > min_delta
    high = g_max - min_delta
    thresh = torch.where(valid_span, torch.minimum(thresh, high), g_max)
    return thresh

def per_channel_loss(pred, target, loss_type='l1'):
    if loss_type == 'l1':
        return (pred - target).abs()
    elif loss_type == 'l2':
        d = pred - target
        return d * d
    elif loss_type == 'huber':
        # smooth_l1_loss 返回已reduction的标量或逐像素；这里要逐像素
        return F.smooth_l1_loss(pred, target, reduction='none')
    else:
        raise ValueError("loss_type must be 'l1' | 'l2' | 'huber'")

def masked_reduce(per, mask=None, eps=1e-8, reduction='mean'):
    """
    per: (B,1,H,W) 或 (B,C,H,W) 的逐像素损失
    mask: (B,1,H,W) 的0/1掩膜; None 表示不掩膜
    reduction: 'mean' | 'sum'
    """
    if mask is not None:
        per = per * mask
        if reduction == 'mean':
            denom = mask.sum().clamp_min(eps)
            return per.sum() / denom
        elif reduction == 'sum':
            return per.sum()
        else:
            return per
    else:
        if reduction == 'mean':
            return per.mean()
        elif reduction == 'sum':
            return per.sum()
        else:
            return per

def edge_guided_loss(
    pred, target,
    thresh_mode='otsu',
    thresh=0.9,
    dilate=1,
    loss_type='l1',
    eps=1e-6,
    normalize_grad=True,
    detach_mask=True,
):
    assert pred.shape == target.shape

    # 1) edge mask from GT grad
    g, _, _ = compute_sobel_grad(target, eps=eps, normalize=normalize_grad)  # (B,1,H,W)

    if thresh_mode == 'fixed':
        T = torch.as_tensor(thresh, device=pred.device, dtype=pred.dtype)
        edge_mask = (g > T).float()
    elif thresh_mode == 'percentile':
        q = torch.quantile(g.flatten(1), torch.as_tensor(thresh, device=g.device), dim=1)
        edge_mask = (g > q.view(-1,1,1,1)).float()
    elif thresh_mode == 'otsu':
        T = batched_otsu_threshold(g, num_bins=256, eps=eps, min_delta=eps)
        edge_mask = (g > T.view(-1,1,1,1)).float()

        # 无边缘保护：当某张图没有边缘像素时，保留其最大梯度像素为边缘
        edge_count = edge_mask.flatten(1).sum(dim=1)
        no_edge = edge_count <= 0
        if no_edge.any():
            flat_g = g.flatten(1)
            max_idx = flat_g.argmax(dim=1, keepdim=True)
            fallback = torch.zeros_like(flat_g)
            fallback.scatter_(1, max_idx, 1.0)
            flat_mask = edge_mask.flatten(1)
            flat_mask[no_edge] = fallback[no_edge]
            edge_mask = flat_mask.view_as(edge_mask)
    else:
        raise ValueError("thresh_mode must be 'fixed' | 'percentile' | 'otsu'")

    if dilate and dilate > 0:
        k = dilate * 2 + 1
        edge_mask = F.max_pool2d(edge_mask, kernel_size=k, stride=1, padding=dilate)

    if detach_mask:
        edge_mask = edge_mask.detach()

    smooth_mask = (1.0 - edge_mask).clamp(0.0, 1.0)

    # 2) pixel loss map (mean over C to avoid channel-count scaling)
    per_pix = per_channel_loss(pred, target, loss_type=loss_type)  # (B,C,H,W) or (B,1,H,W)
    if per_pix.dim() == 4 and per_pix.shape[1] > 1:
        per_pix = per_pix.mean(dim=1, keepdim=True)                # -> (B,1,H,W)

    edge_pix = masked_reduce(per_pix, edge_mask,  eps=eps, reduction='mean')
    smooth_pix = masked_reduce(per_pix, smooth_mask, eps=eps, reduction='mean')

    # 3) grad loss (only on edge)
    _, gpx, gpy = compute_sobel_grad(pred,   eps=eps, normalize=normalize_grad)
    _, gtx, gty = compute_sobel_grad(target, eps=eps, normalize=normalize_grad)
    grad_diff = (gpx - gtx).abs() + (gpy - gty).abs()
    edge_grad = masked_reduce(grad_diff, edge_mask, eps=eps, reduction='mean')

    # exp1: keep the original edge/smooth split, but use explicit weights.
    # The previous dynamic ratios forced all three terms to the same scale,
    # which effectively removed control over the gradient and smooth losses.
    total = (
        1.0 * smooth_pix
        + 2.0 * edge_pix
        + 0.2 * edge_grad
    )
    return total

if os.name == "nt":
    data_root = r"D:\nuoqi\S2R-HDR\data\20260818"
    base_experiment_root = r"D:\nuoqi\halo-free\Decoupled_Learning\20260815_edgepreserving"
else:
    data_root = "/mnt/d/nuoqi/S2R-HDR/data/20260818"
    base_experiment_root = "/mnt/d/nuoqi/halo-free/Decoupled_Learning/20260815_edgepreserving"
experiment_root = os.path.join(base_experiment_root, "experiments", "exp4", "exp4B")
downsample_folder_train = os.path.join(data_root, "traininput-l0p500-a1p400_downsample")
origin_input_train = os.path.join(data_root, "traininput-l0p500-a1p400")
input_folder_train_label = os.path.join(data_root, "trainlabel-l0p500-a1p400")

downsample_folder_test = os.path.join(data_root, "testinput-l0p500-a1p400-200_downsample")
origin_input_test = os.path.join(data_root, "testinput-l0p500-a1p400-200")
input_folder_test_label = os.path.join(data_root, "testlabel-l0p500-a1p400-200")

##### - 数据加载 - #######
class GuidedFilterDataset(Dataset):
    def __init__(self, x_lr_dir, origin_input_dir, label_dir):
        self.x_lr_files = sorted([os.path.join(x_lr_dir, f) for f in os.listdir(x_lr_dir)])
        self.origin_input_files = sorted([os.path.join(origin_input_dir, f) for f in os.listdir(origin_input_dir)])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        print(f"LR: {len(self.x_lr_files)}, HR: {len(self.origin_input_files)}, Label: {len(self.label_files)}")

    def __len__(self):
        return min(len(self.x_lr_files), len(self.origin_input_files), len(self.label_files))

    def __getitem__(self, idx):
        x_lr = cv2.imread(self.x_lr_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        origin_input   = cv2.imread(self.origin_input_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        label          = cv2.imread(self.label_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        input_hr_h, input_hr_w = origin_input.shape
        input_lr_h, input_lr_w = x_lr.shape

        return {
            'input_lr': torch.from_numpy(x_lr).unsqueeze(0).unsqueeze(0),
            'origin_input_hr': torch.from_numpy(origin_input).unsqueeze(0).unsqueeze(0),
            'target_img': torch.from_numpy(label).unsqueeze(0).unsqueeze(0),
            'input_height_hr': input_hr_h,
            'input_width_hr': input_hr_w,
            'input_height_lr': input_lr_h,
            'input_width_lr': input_lr_w,
        }

def var_custom_collate(batch):
    min_h_hr = min([item['input_height_hr'] for item in batch])
    min_w_hr = min([item['input_width_hr'] for item in batch])
    min_h_lr = min([item['input_height_lr'] for item in batch])
    min_w_lr = min([item['input_width_lr'] for item in batch])

    batch_input_lr = torch.zeros(len(batch), 1, min_h_lr, min_w_lr)
    batch_input_origin   = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_target_images = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)

    for idx, item in enumerate(batch):
        # 直接裁切到 min_h, min_w
        input_origin_crop = item['origin_input_hr'][:, :, :min_h_hr, :min_w_hr]
        input_lr_crop = item['input_lr'][:, :, :min_h_lr, :min_w_lr]
        label_crop  = item['target_img'][:, :, :min_h_hr, :min_w_hr]

        # 放到 batch
        batch_input_origin[idx] = input_origin_crop
        batch_input_lr[idx] = input_lr_crop
        batch_target_images[idx] = label_crop

    return batch_input_lr, batch_input_origin, batch_target_images

##### 构建训练集和测试集 ####
train_dataset = GuidedFilterDataset(
    x_lr_dir = downsample_folder_train,
    origin_input_dir = origin_input_train,
    label_dir = input_folder_train_label
)

test_dataset = GuidedFilterDataset(
    x_lr_dir = downsample_folder_test,
    origin_input_dir = origin_input_test,
    label_dir = input_folder_test_label
)

def crop_to_even(tensor):
    """
    将输入 Tensor 裁剪到高宽为偶数，保持左上角对齐（裁掉底部和右边）。
    tensor: [B, C, H, W]
    返回裁剪后的 tensor
    """
    _, _, H, W = tensor.shape
    new_H = H if H % 2 == 0 else H - 1
    new_W = W if W % 2 == 0 else W - 1
    return tensor[:, :, :new_H, :new_W]

visible_gpu_count = max(1, len([gpu for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if gpu]))
train_loader = DataLoader(train_dataset, batch_size=visible_gpu_count, shuffle=True, num_workers=0,
                          collate_fn=var_custom_collate, pin_memory=True, drop_last=True)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                          collate_fn=var_custom_collate, pin_memory=True, drop_last=False)

# --- 在每个 epoch 结束后保存 ---
def save_checkpoint(epoch, best_loss, is_best=False):
    # 保存完整状态，方便续训
    # full_path = os.path.join(save_dir, "refine_filter_full_mix_v12.pth")
    # torch.save({
    #     'epoch': epoch + 1,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'scheduler_state_dict': scheduler.state_dict(),
    #     'best_loss': best_loss
    # }, full_path)
    # print(f"Checkpoint saved: {full_path}")

    # # 保存纯模型 (方便推理或部署)
    # last_model_path = os.path.join(save_dir, "refine_filter_last_mix_v12.pth")
    # torch.save(model.state_dict(), last_model_path)
    # print(f"Last model weights saved: {last_model_path}")

    # 如果是最佳模型，另存一份
    if is_best:
        best_model_path = os.path.join(save_dir, "rf_3stage_best_20260818_l0p500_a1p400.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"*** Best model updated: {best_model_path} (Test MSE: {best_loss:.6f}) ***")

def save_batch_images(tensor, save_dir, prefix="epoch", epoch=0, batch_idx=0):
    os.makedirs(save_dir, exist_ok=True)
    tensor = tensor.detach().cpu().numpy()   # (B,1,H,W)

    for i in range(tensor.shape[0]):
        img = tensor[i,0]   # (H,W)

        # 找到非零区域
        mask_nonzero = img > 0
        if mask_nonzero.any():
            vals = img[mask_nonzero]
            vmin, vmax = vals.min(), vals.max()
            # 只缩放非零区域
            img_norm = np.zeros_like(img, dtype=np.float32)
            if vmax > vmin:
                img_norm[mask_nonzero] = (img[mask_nonzero] - vmin) / (vmax - vmin)
        else:
            img_norm = np.zeros_like(img, dtype=np.float32)

        img_uint8 = (img *255).astype(np.uint8)
        #img_uint8 = (img_norm * 255).astype(np.uint8)

        filename = os.path.join(
            save_dir, f"{prefix}{epoch}_batch{batch_idx}_sample{i}.png"
        )
        imageio.imwrite(filename, img_uint8)
        print(f"[Saved] {filename}")


##### tensorboard 启动指令 #######
task_name = "exp4B_groupnorm_deep_supervision_20260818_l0p500_a1p400"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(
    experiment_root,
    "logs",
    f"{task_name}_{timestamp}",
)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

######### psnr 计算 ###########
def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1 / mse)
    return psnr, mse

device = torch.device('cuda:0')
stage1_model_path = os.path.join(
    base_experiment_root,
    "models",
    "convguided_20260818_l0p500_a1p400",
    "convguided_filter_best_20260818_l0p500_a1p400.pth",
)
stage1_model = load_frozen_stage1(stage1_model_path, device)
model = CascadeRefine3Stage(base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True)
model = torch.nn.DataParallel(model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
best_loss = float("inf")
save_dir = os.path.join(experiment_root, "models", "exp4B_groupnorm_deep_supervision")
os.makedirs(save_dir, exist_ok=True)
epochs = 80

if __name__ == "__main__":
#def train_one_epoch():
######## -- 模型训练 -- #############
    for epoch in tqdm(range(epochs), desc="Training Progress", dynamic_ncols=True):
        model.train()
        epoch_mse = 0
        epoch_psnr = 0
        train_progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"RF train {epoch + 1}/{epochs}",
            leave=True,
            mininterval=1.0,
            dynamic_ncols=True,
        )
        for batch_idx, (batch_input_lr, batch_input_origin, batch_label) in train_progress:
            torch.cuda.synchronize(); t0 = time.time()

            batch_input_lr = crop_to_even(batch_input_lr)
            batch_input_origin = crop_to_even(batch_input_origin)
            batch_label        = crop_to_even(batch_label)
            torch.cuda.synchronize(); t1 = time.time()

            batch_input_lr = batch_input_lr.to(device)
            batch_input_origin = batch_input_origin.to(device)
            batch_label = batch_label.to(device)

            # Generate the frozen first-stage coarse image online for this batch.
            with torch.no_grad():
                batch_input = stage1_model(batch_input_lr - 0.5, batch_input_origin - 0.5)
            batch_input_origin = batch_input_origin - 0.5
            batch_input = batch_input - 0.5
            batch_label        = batch_label - 0.5

            # 前向
            outputs = model(batch_input_origin, batch_input)
            torch.cuda.synchronize(); t3 = time.time()

            loss, (l1, l2, l3) = cascade_loss(
                outputs, batch_input, batch_label,
                thresh_mode='otsu',
                dilate=0,
                loss_type='l1',
                normalize_grad=True
            )

            # ----- 保存mask结果看看是否符合预期 ------ #
            # save_output_dir = 'D:/nuoqi/halo-free/Decoupled_Learning/HDRv1/train_vis/refinev6'
            # if batch_idx == len(train_loader) // 2:
            #     save_batch_images(outputs["final"], save_output_dir, prefix="pred", epoch=epoch, batch_idx=batch_idx)

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(); t4 = time.time()

            output_final = outputs["final"] + 0.5
            output_final = torch.clamp(output_final, 0, 1)
            batch_label = batch_label + 0.5
            psnr, mse = calc_psnr(output_final.detach(), batch_label)
            if batch_idx % 200 == 0:
                print(f"iter MSE loss: {mse:.6f}")
            epoch_mse += mse
            epoch_psnr += psnr
            train_progress.set_postfix(mse=f"{mse:.6f}", psnr=f"{psnr:.2f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")
            # 每 50 step 记录到 TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 50 == 0:
                writer.add_scalar('train/mse', mse, global_step)
                writer.add_scalar('train/psnr', psnr, global_step)
                writer.add_scalar('train/l1', l1.item(), global_step)
                writer.add_scalar('train/l2', l2.item(), global_step)
                writer.add_scalar('train/l3', l3.item(), global_step)
        avg_epoch_mse = epoch_mse / len(train_loader)
        avg_epoch_psnr = epoch_psnr / len(train_loader)
            # 打印本 epoch 的平均指标
        print(f"Epoch {epoch+1}/{epochs} "
            f"- MSE: {avg_epoch_mse:.6f} "
            f"- PSNR: {avg_epoch_psnr:.2f} dB ")
        writer.add_scalar('train/mse_epoch', avg_epoch_mse, epoch)
        writer.add_scalar('train/psnr_epoch', avg_epoch_psnr, epoch)
        scheduler.step()

# ---------------- 测试集验证阶段 ----------------
        model.eval()
        test_mse = 0
        test_psnr = 0
        with torch.no_grad():
            test_progress = tqdm(
                test_loader,
                total=len(test_loader),
                desc=f"RF test {epoch + 1}/{epochs}",
                leave=True,
                mininterval=1.0,
                dynamic_ncols=True,
            )
            for test_input_lr, test_input_origin, test_label in test_progress:
                test_input_lr = crop_to_even(test_input_lr)
                test_input_origin = crop_to_even(test_input_origin)
                test_label = crop_to_even(test_label)

                test_input_lr = test_input_lr.to(device)
                test_input_origin = test_input_origin.to(device)
                test_label = test_label.to(device)

                test_input = stage1_model(test_input_lr - 0.5, test_input_origin - 0.5)
                test_input_origin = test_input_origin - 0.5
                test_input = test_input - 0.5
                test_label = test_label - 0.5
                test_outputs = model(test_input_origin, test_input)

                loss, _ = cascade_loss(
                    test_outputs, test_input, test_label,
                    thresh_mode='otsu',
                    dilate=0,
                    loss_type='l1',
                    normalize_grad=True
                )

                test_output = test_outputs["final"] + 0.5
                test_output = torch.clamp(test_output, 0, 1)
                test_label  = test_label + 0.5
                psnr, mse   = calc_psnr(test_output, test_label)
                test_mse += mse
                test_psnr += psnr
                test_progress.set_postfix(mse=f"{mse:.6f}", psnr=f"{psnr:.2f}")

        avg_test_mse = test_mse / len(test_loader)
        avg_test_psnr = test_psnr / len(test_loader)

        print(f"Epoch {epoch+1}/{epochs} "
            f"- Test MSE: {avg_test_mse:.6f} "
            f"- Test PSNR: {avg_test_psnr:.2f} dB")
        # --- 记录到 TensorBoard ---
        writer.add_scalar('test/mse', avg_test_mse, epoch)
        writer.add_scalar('test/psnr', avg_test_psnr, epoch)

        # 判断是否最佳模型
        is_best = avg_test_mse < best_loss
        if is_best:
            best_loss = avg_test_mse

        # 保存模型
        save_checkpoint(epoch, best_loss, is_best=is_best)

    # 关闭 TensorBoard writer
    writer.close()
