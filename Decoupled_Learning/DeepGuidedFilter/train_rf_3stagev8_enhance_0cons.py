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

def make_norm(num_channels, norm_type="batch", gn_groups=8):
    norm = norm_type.lower()
    if norm in ("batch", "bn", "batchnorm"):
        return nn.BatchNorm2d(num_channels)
    if norm in ("group", "gn", "groupnorm"):
        groups = min(gn_groups, num_channels)
        while groups > 1 and (num_channels % groups) != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")

class Stage1DeltaNet(nn.Module):
    def __init__(
        self,
        in_channels=5,
        base_channels=32,
        delta_scale=0.40,
        sign_prior_scale=0.0,
        norm_type="batch",
        gn_groups=8,
    ):
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.sign_prior_scale = float(sign_prior_scale)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            make_norm(16, norm_type=norm_type, gn_groups=gn_groups),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, base_channels, kernel_size=3, padding=1, bias=False),
            make_norm(base_channels, norm_type=norm_type, gn_groups=gn_groups),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            make_norm(base_channels, norm_type=norm_type, gn_groups=gn_groups),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, 2, kernel_size=1, bias=True)
        )

    def forward(self, x_raw, x_coarse):
        _, gx, gy = compute_sobel_grad(x_coarse, normalize=False)
        gx = gx / (gx.abs().amax(dim=(2, 3), keepdim=True) + 1e-6)
        gy = gy / (gy.abs().amax(dim=(2, 3), keepdim=True) + 1e-6)
        raw_delta = x_raw - x_coarse
        inp = torch.cat([x_raw, x_coarse, raw_delta, gx, gy], dim=1)

        stage1_raw = self.net(inp)
        mag_raw = stage1_raw[:, 0:1]
        sign_raw = stage1_raw[:, 1:2]

        # Keep train/infer consistent: do not inject GT-unavailable strong sign bias in forward.
        # If ablation is needed, set sign_prior_scale > 0.
        if self.sign_prior_scale != 0.0:
            sign_logits = sign_raw + self.sign_prior_scale * raw_delta
        else:
            sign_logits = sign_raw
        sign_map = torch.tanh(sign_logits)
        mag_map = self.delta_scale * torch.sigmoid(mag_raw)
        delta = sign_map * mag_map
        return {
            "delta": delta,
            "sign_logits": sign_logits,
            "sign_map": sign_map,
            "mag_map": mag_map,
        }

class ABAdjustNet(nn.Module):
    def __init__(
        self,
        base_channels=32,
        a_scale=0.5,
        b_scale=0.4,
        use_delta_as_input=True,
        norm_type="batch",
        gn_groups=8,
    ):
        super().__init__()
        self.a_scale = a_scale
        self.b_scale = b_scale
        self.use_delta_as_input = use_delta_as_input

        in_ch = 3 + (1 if use_delta_as_input else 0)  # raw + coarse_k + grad + (delta_prev可选)

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
            make_norm(16, norm_type=norm_type, gn_groups=gn_groups),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, base_channels, 3, padding=1, bias=False),
            make_norm(base_channels, norm_type=norm_type, gn_groups=gn_groups),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            make_norm(base_channels, norm_type=norm_type, gn_groups=gn_groups),
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
    def __init__(
        self,
        base_channels=32,
        delta_scale=0.40,
        a_scale=0.5,
        b_scale=0.4,
        use_delta_as_input=True,
        norm_type="batch",
        gn_groups=8,
    ):
        super().__init__()
        self.stage1 = Stage1DeltaNet(
            base_channels=base_channels,
            delta_scale=delta_scale,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )
        self.stage2 = ABAdjustNet(base_channels=base_channels, a_scale=a_scale, b_scale=b_scale,
                                 use_delta_as_input=use_delta_as_input,
                                 norm_type=norm_type, gn_groups=gn_groups)
        self.stage3 = ABAdjustNet(base_channels=base_channels, a_scale=a_scale, b_scale=b_scale,
                                 use_delta_as_input=use_delta_as_input,
                                 norm_type=norm_type, gn_groups=gn_groups)

    def forward(self, x_raw, x_coarse0):
        # ---- Stage 1 ----
        stage1_out = self.stage1(x_raw, x_coarse0)
        delta1 = stage1_out["delta"]                      # refine delta
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
            "stage1_sign_logits": stage1_out["sign_logits"],
            "stage1_sign_map": stage1_out["sign_map"],
            "stage1_mag_map": stage1_out["mag_map"],
            "a2": a2, "b2": b2,
            "a3": a3, "b3": b3,
        }

def cascade_loss(outputs, x_coarse0, gt,
                 thresh_mode='otsu', thresh=0.6, dilate=0,
                 loss_type='l1', normalize_grad=True,
                 precomputed_edge_mask=None,
                 precomputed_smooth_mask=None,
                 tau_zero=3e-8,
                 sign_small_tau=2e-2,
                 sign_small_gain=2.0,
                 sign_smooth_gain=1.0,
                 stage1_sign_weight=6.0,
                 sign_lambda_pn=1.0,
                 sign_lambda_zero_delta=1.0,
                 stage1_edge_weight=2.0,
                 stage2_weight=0.7,
                 stage3_weight=1.0,
                 final_weight=0.5,
                 dynamic_balance=True,
                 dynamic_clip_min=0.1,
                 dynamic_clip_max=10.0,
                 dynamic_eps=1e-6):
    delta_gt = gt - x_coarse0

    l1 = edge_guided_loss(outputs["delta1"],  delta_gt,
                          thresh_mode=thresh_mode, thresh=thresh, dilate=dilate,
                          loss_type=loss_type, normalize_grad=normalize_grad,
                          precomputed_edge_mask=precomputed_edge_mask,
                          precomputed_smooth_mask=precomputed_smooth_mask)
    l2 = edge_guided_loss(outputs["delta2"], delta_gt,
                          thresh_mode=thresh_mode, thresh=thresh, dilate=dilate,
                          loss_type=loss_type, normalize_grad=normalize_grad,
                          precomputed_edge_mask=precomputed_edge_mask,
                          precomputed_smooth_mask=precomputed_smooth_mask)
    l3 = edge_guided_loss(outputs["delta3"], delta_gt,
                          thresh_mode=thresh_mode, thresh=thresh, dilate=dilate,
                          loss_type=loss_type, normalize_grad=normalize_grad,
                          precomputed_edge_mask=precomputed_edge_mask,
                          precomputed_smooth_mask=precomputed_smooth_mask)
    sign_weight_pn = build_sign_weight_map(
        delta_gt,
        smooth_mask=precomputed_smooth_mask,
        small_focus_tau=sign_small_tau,
        small_focus_gain=sign_small_gain,
        smooth_focus_gain=sign_smooth_gain,
    )
    mask_pn, mask_zero = build_sign_region_masks(
        delta_gt,
        tau_zero=tau_zero,
    )
    stage1_sign_pn, stage1_sign_zero_logits, stage1_sign_zero_delta = two_stage_sign_loss(
        outputs["stage1_sign_logits"],
        outputs["delta1"],
        delta_gt,
        mask_pn=mask_pn,
        mask_zero=mask_zero,
        weight_pn=sign_weight_pn,
    )
    stage1_sign = (
        sign_lambda_pn * stage1_sign_pn
        + sign_lambda_zero_delta * stage1_sign_zero_delta
    )
    final_l1 = F.l1_loss(outputs["final"], gt)
    t_stage1 = stage1_edge_weight * l1
    t_stage2 = stage2_weight * l2
    t_stage3 = stage3_weight * l3
    t_sign = stage1_sign_weight * stage1_sign
    t_final = final_weight * final_l1

    if dynamic_balance:
        # v3-style dynamic alignment: keep different terms on similar scales
        # so sign loss cannot dominate and suppress residual amplitude learning.
        ref = t_stage3.detach().clamp_min(dynamic_eps)

        def align(term):
            ratio = (term / ref).detach().clamp(min=dynamic_clip_min, max=dynamic_clip_max)
            return term / ratio, ratio

        stage1_aligned, r_stage1 = align(t_stage1)
        stage2_aligned, r_stage2 = align(t_stage2)
        sign_aligned, r_sign = align(t_sign)
        final_aligned, r_final = align(t_final)
        total = t_stage3 + stage1_aligned + stage2_aligned + sign_aligned + final_aligned
    else:
        total = t_stage1 + t_stage2 + t_stage3 + t_sign + t_final

    aux = {
        "stage1_edge": l1,
        "stage2_edge": l2,
        "stage3_edge": l3,
        "stage1_sign": stage1_sign,
        "stage1_sign_pn": stage1_sign_pn,
        "stage1_sign_zero_logits": stage1_sign_zero_logits,
        "stage1_sign_zero_delta": stage1_sign_zero_delta,
        "final_l1": final_l1,
        "term_stage1": t_stage1,
        "term_stage2": t_stage2,
        "term_stage3": t_stage3,
        "term_sign": t_sign,
        "term_final": t_final,
    }
    if dynamic_balance:
        aux.update({
            "dyn_ratio_stage1": r_stage1,
            "dyn_ratio_stage2": r_stage2,
            "dyn_ratio_sign": r_sign,
            "dyn_ratio_final": r_final,
        })
    return total, aux

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

def build_edge_mask_from_grad(
    g,
    thresh_mode='otsu',
    thresh=0.9,
    dilate=1,
    eps=1e-6,
):
    if thresh_mode == 'fixed':
        T = torch.as_tensor(thresh, device=g.device, dtype=g.dtype)
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

    return edge_mask

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

def build_sign_region_masks(target, tau_zero=3e-8):
    abs_target = target.abs()
    mask_pn = (abs_target > tau_zero).float()
    mask_zero = (abs_target <= tau_zero).float()
    return mask_pn, mask_zero

def build_sign_weight_map(
    target,
    smooth_mask=None,
    small_focus_tau=2e-2,
    small_focus_gain=2.0,
    smooth_focus_gain=1.0,
    eps=1e-8,
):
    weight = torch.ones_like(target)
    if smooth_mask is not None:
        weight = weight + smooth_focus_gain * smooth_mask
    if small_focus_gain > 0:
        small_weight = torch.exp(-target.abs() / (small_focus_tau + eps))
        weight = weight * (1.0 + small_focus_gain * small_weight)
    return weight

def sign_consistency_loss(sign_logits, target, mask=None, weight=None, eps=1e-8):
    sign_target = torch.where(target > 0, torch.ones_like(target), -torch.ones_like(target))
    # Logistic sign loss on {-1, +1}.
    signed_margin = sign_target * sign_logits
    loss_map = F.softplus(-signed_margin)
    if weight is not None:
        loss_map = loss_map * weight
    return masked_reduce(loss_map, mask=mask, eps=eps, reduction='mean')

def two_stage_sign_loss(sign_logits, delta_pred, target, mask_pn, mask_zero, weight_pn=None, eps=1e-8):
    sign_target = torch.where(target > 0, torch.ones_like(target), -torch.ones_like(target))

    # Segment 1: |target| > tau_zero, keep positive/negative sign supervision.
    signed_margin = sign_target * sign_logits
    sign_pn_map = F.softplus(-signed_margin)
    if weight_pn is not None:
        sign_pn_map = sign_pn_map * weight_pn
    sign_pn = masked_reduce(sign_pn_map, mask=mask_pn, eps=eps, reduction='mean')

    # Segment 2: |target| <= tau_zero, only force the predicted residual toward zero.
    zero_delta = F.smooth_l1_loss(delta_pred, torch.zeros_like(delta_pred), reduction='none')
    sign_zero_logits = sign_logits.new_zeros(())
    sign_zero_delta = masked_reduce(zero_delta, mask=mask_zero, eps=eps, reduction='mean')
    return sign_pn, sign_zero_logits, sign_zero_delta

def edge_guided_loss(
    pred, target,
    thresh_mode='otsu',
    thresh=0.9,
    dilate=1,
    loss_type='l1',
    eps=1e-6,
    normalize_grad=True,
    detach_mask=True,
    w_dir=0.3,
    w_smooth_pix=0.6,
    precomputed_edge_mask=None,
    precomputed_smooth_mask=None,
):
    assert pred.shape == target.shape

    # 1) edge mask from GT grad (or precomputed)
    if precomputed_edge_mask is None:
        g, _, _ = compute_sobel_grad(target, eps=eps, normalize=normalize_grad)  # (B,1,H,W)
        edge_mask = build_edge_mask_from_grad(
            g, thresh_mode=thresh_mode, thresh=thresh, dilate=dilate, eps=eps
        )
    else:
        edge_mask = precomputed_edge_mask

    if detach_mask:
        edge_mask = edge_mask.detach()

    if precomputed_smooth_mask is None:
        smooth_mask = (1.0 - edge_mask).clamp(0.0, 1.0)
    else:
        smooth_mask = precomputed_smooth_mask

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

    # 4) edge direction consistency (only on edge)
    pred_mag = torch.sqrt(gpx * gpx + gpy * gpy + eps)
    gt_mag = torch.sqrt(gtx * gtx + gty * gty + eps)
    dir_px = gpx / pred_mag
    dir_py = gpy / pred_mag
    dir_tx = gtx / gt_mag
    dir_ty = gty / gt_mag
    dir_diff = 1.0 - (dir_px * dir_tx + dir_py * dir_ty)
    edge_dir = masked_reduce(dir_diff, edge_mask, eps=eps, reduction='mean')

    # 5) dynamic align: align each aux to edge_pix
    #    Add smooth pixel supervision to avoid an overly conservative smooth-area solution.
    ratio_edge = (edge_grad / (edge_pix + eps)).detach().clamp_min(eps)
    ratio_smooth_pix = (smooth_pix / (edge_pix + eps)).detach().clamp_min(eps)
    total = edge_pix \
        + edge_grad  / ratio_edge \
        + w_dir * edge_dir \
        + w_smooth_pix * (smooth_pix / ratio_smooth_pix)

    return total

###### 训练输入 #############
origin_input_train = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/traininput'
input_folder_train = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/train_refine_input'
input_folder_train_label = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/trainlabel'

###### 测试输入 ##############
origin_input_test = '/mnt/d/nuoqi/S2R-HDR/data/Single-Image-HDR/testinput'
input_folder_test = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/test_refine_input'
input_folder_test_label = '/mnt/d/nuoqi/S2R-HDR/data/Single-Image-HDR/testlabel'

##### - 数据加载 - #######
class GuidedFilterDataset(Dataset):
    def __init__(self, origin_input_dir, input_dir, label_dir):
        self.origin_input_files = sorted([os.path.join(origin_input_dir, f) for f in os.listdir(origin_input_dir)])
        self.input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        print(f"INPUT: {len(self.input_files)}, Label: {len(self.label_files)}")

    def __len__(self):
        return min(len(self.input_files), len(self.label_files))

    def __getitem__(self, idx):
        origin_input   = cv2.imread(self.origin_input_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        input          = cv2.imread(self.input_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        label          = cv2.imread(self.label_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        input_hr_h, input_hr_w = input.shape

        return {
            'origin_input_hr': torch.from_numpy(origin_input).unsqueeze(0).unsqueeze(0),
            'input_hr': torch.from_numpy(input).unsqueeze(0).unsqueeze(0),
            'target_img': torch.from_numpy(label).unsqueeze(0).unsqueeze(0),
            'input_height_hr': input_hr_h,
            'input_width_hr': input_hr_w,
        }

def var_custom_collate(batch):
    min_h_hr = min([item['input_height_hr'] for item in batch])
    min_w_hr = min([item['input_width_hr'] for item in batch])

    batch_input = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_input_origin   = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_target_images = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)

    for idx, item in enumerate(batch):
        # 直接裁切到 min_h, min_w
        input_origin_crop = item['origin_input_hr'][:, :, :min_h_hr, :min_w_hr]
        input_crop   = item['input_hr'][:, :, :min_h_hr, :min_w_hr]
        label_crop  = item['target_img'][:, :, :min_h_hr, :min_w_hr]

        # 放到 batch
        batch_input_origin[idx] = input_origin_crop
        batch_input[idx] = input_crop
        batch_target_images[idx] = label_crop

    return batch_input_origin, batch_input, batch_target_images

##### 构建训练集和测试集 ####
train_dataset = GuidedFilterDataset(
    origin_input_dir = origin_input_train,
    input_dir = input_folder_train,
    label_dir = input_folder_train_label
)

test_dataset = GuidedFilterDataset(
    origin_input_dir = origin_input_test,
    input_dir = input_folder_test,
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

train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=12,
                          collate_fn=var_custom_collate, pin_memory=False, drop_last=True)

test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4,
                          collate_fn=var_custom_collate, pin_memory=False, drop_last=False)

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
        best_model_path = os.path.join(save_dir, f"rf_3stage_best_gtmask_signloss_v8_enhance_0cons_optv2.pth")
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
task_name = "rf_3stage_gtmask_signloss_v8_enhance_0cons_optv2"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("/mnt/d/nuoqi/halo-free/Decoupled_Learning/DeepGuidedFilter/log_dir", f"{task_name}_{timestamp}")
writer = SummaryWriter(log_dir=log_dir)

######### psnr 计算 ###########
def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1 / mse)
    return psnr, mse

norm_type = "group"
gn_groups = 8
delta_scale = 0.40
a_scale = 0.5
b_scale = 0.4

def create_model():
    return CascadeRefine3Stage(
        base_channels=32,
        a_scale=a_scale,
        b_scale=b_scale,
        use_delta_as_input=True,
        norm_type=norm_type,
        gn_groups=gn_groups,
        delta_scale=delta_scale,
    )

device_ids = [0, 1]
device = torch.device(f'cuda:{device_ids[0]}')
model = create_model()
visible_gpus = torch.cuda.device_count()
if visible_gpus <= max(device_ids):
    raise RuntimeError(
        f"Requested GPUs {device_ids}, but only {visible_gpus} CUDA device(s) are visible. "
        "Please set CUDA_VISIBLE_DEVICES correctly."
    )
model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0]).to(device)
print(f"[Info] DataParallel device_ids={device_ids}")
print(f"[Info] Normalization: {norm_type} (groups={gn_groups})")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
best_loss = float("inf")
save_dir = "/mnt/d/nuoqi/halo-free/Decoupled_Learning/DeepGuidedFilter/models_retrain"
epochs = 80

if __name__ == "__main__":
#def train_one_epoch():
######## -- 模型训练 -- #############
    for epoch in tqdm(range(epochs), desc="Training Progress", dynamic_ncols=True):
        model.train()
        epoch_mse = 0
        epoch_psnr = 0
        for batch_idx, (batch_input_origin, batch_input, batch_label) in enumerate(train_loader):
            torch.cuda.synchronize()
            t0 = time.time()

            batch_input_origin = crop_to_even(batch_input_origin)
            batch_input        = crop_to_even(batch_input)
            batch_label        = crop_to_even(batch_label)
            torch.cuda.synchronize()
            t1 = time.time()

            batch_input_origin = batch_input_origin.to(device)
            batch_input = batch_input.to(device)
            batch_label = batch_label.to(device)

            # ----- 输入网络进行处理 ------ #
            batch_input_origin = batch_input_origin - 0.5
            batch_input        = batch_input - 0.5       # coarse filtered img
            batch_label        = batch_label - 0.5

            # 前向
            outputs = model(batch_input_origin, batch_input)
            torch.cuda.synchronize()
            t3 = time.time()

            # Precompute GT-delta edge mask on GPU, consistent with v3.
            with torch.no_grad():
                delta_gt = batch_label - batch_input
                g, _, _ = compute_sobel_grad(delta_gt, eps=1e-6, normalize=True)
                batch_edge_mask = build_edge_mask_from_grad(
                    g, thresh_mode='otsu', thresh=0.9, dilate=0, eps=1e-6
                )
                batch_smooth_mask = (1.0 - batch_edge_mask).clamp(0.0, 1.0)

            loss, aux_losses = cascade_loss(
                outputs, batch_input, batch_label,
                thresh_mode='otsu',
                dilate=0,
                loss_type='l1',
                normalize_grad=True,
                precomputed_edge_mask=batch_edge_mask,
                precomputed_smooth_mask=batch_smooth_mask
            )

            # ----- 保存mask结果看看是否符合预期 ------ #
            # save_output_dir = 'D:/nuoqi/halo-free/Decoupled_Learning/HDRv1/train_vis/refinev6'
            # if batch_idx == len(train_loader) // 2:
            #     save_batch_images(outputs["final"], save_output_dir, prefix="pred", epoch=epoch, batch_idx=batch_idx)

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            t4 = time.time()

            output_final = outputs["final"] + 0.5
            output_final = torch.clamp(output_final, 0, 1)
            batch_label = batch_label + 0.5
            psnr, mse = calc_psnr(output_final.detach(), batch_label)
            if batch_idx % 200 == 0:
                print(f"iter MSE loss: {mse:.6f}")
            epoch_mse += mse
            epoch_psnr += psnr
            # 每 50 step 记录到 TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 50 == 0:
                writer.add_scalar('train/mse', mse, global_step)
                writer.add_scalar('train/psnr', psnr, global_step)
                writer.add_scalar('train/stage1_edge', aux_losses['stage1_edge'].item(), global_step)
                writer.add_scalar('train/stage2_edge', aux_losses['stage2_edge'].item(), global_step)
                writer.add_scalar('train/stage3_edge', aux_losses['stage3_edge'].item(), global_step)
                writer.add_scalar('train/stage1_sign', aux_losses['stage1_sign'].item(), global_step)
                writer.add_scalar('train/final_l1', aux_losses['final_l1'].item(), global_step)
                if 'term_sign' in aux_losses:
                    writer.add_scalar('train_weighted/term_stage1', aux_losses['term_stage1'].item(), global_step)
                    writer.add_scalar('train_weighted/term_stage2', aux_losses['term_stage2'].item(), global_step)
                    writer.add_scalar('train_weighted/term_stage3', aux_losses['term_stage3'].item(), global_step)
                    writer.add_scalar('train_weighted/term_sign', aux_losses['term_sign'].item(), global_step)
                    writer.add_scalar('train_weighted/term_final', aux_losses['term_final'].item(), global_step)
                if 'dyn_ratio_sign' in aux_losses:
                    writer.add_scalar('train_dynamic_ratio/stage1', aux_losses['dyn_ratio_stage1'].item(), global_step)
                    writer.add_scalar('train_dynamic_ratio/stage2', aux_losses['dyn_ratio_stage2'].item(), global_step)
                    writer.add_scalar('train_dynamic_ratio/sign', aux_losses['dyn_ratio_sign'].item(), global_step)
                    writer.add_scalar('train_dynamic_ratio/final', aux_losses['dyn_ratio_final'].item(), global_step)
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
            for test_input_origin, test_input, test_label in test_loader:
                test_input_origin = crop_to_even(test_input_origin)
                test_input = crop_to_even(test_input)
                test_label = crop_to_even(test_label)

                test_input_origin = test_input_origin.to(device)
                test_input = test_input.to(device)
                test_label = test_label.to(device)

                # ----- 对输入网络的数据进行处理 ----- #
                test_input_origin = test_input_origin - 0.5
                test_input = test_input - 0.5
                test_label = test_label - 0.5
                test_outputs = model(test_input_origin, test_input)

                with torch.no_grad():
                    delta_gt = test_label - test_input
                    g, _, _ = compute_sobel_grad(delta_gt, eps=1e-6, normalize=True)
                    test_edge_mask = build_edge_mask_from_grad(
                        g, thresh_mode='otsu', thresh=0.9, dilate=0, eps=1e-6
                    )
                    test_smooth_mask = (1.0 - test_edge_mask).clamp(0.0, 1.0)

                loss, _ = cascade_loss(
                    test_outputs, test_input, test_label,
                    thresh_mode='otsu',
                    dilate=0,
                    loss_type='l1',
                    normalize_grad=True,
                    precomputed_edge_mask=test_edge_mask,
                    precomputed_smooth_mask=test_smooth_mask
                )

                test_output = test_outputs["final"] + 0.5
                test_output = torch.clamp(test_output, 0, 1)
                test_label  = test_label + 0.5
                psnr, mse   = calc_psnr(test_output, test_label)
                test_mse += mse
                test_psnr += psnr

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
