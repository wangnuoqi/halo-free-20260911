import os
import time
import imageio
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_LOWFREQ_KERNEL = 5
DEFAULT_LOWFREQ_SIGMA = 1.0


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


origin_input_test = "/mnt/d/nuoqi/S2R-HDR/data/Single-Image-HDR/testinput"
input_folder_test = "/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/test_refine_input"
input_folder_test_label = "/mnt/d/nuoqi/S2R-HDR/data/Single-Image-HDR/testlabel"

# origin_input_test = '/mnt/d/nuoqi/S2R-HDR/data/test_vivo/testinput'
# input_folder_test = '/mnt/d/nuoqi/S2R-HDR/data/test_vivo/rf1stage_output'
# input_folder_test_label = '/mnt/d/nuoqi/S2R-HDR/data/test_vivo/testlabel'

model_path = "/mnt/d/nuoqi/halo-free/Decoupled_Learning/DeepGuidedFilter/models_retrain/rf_3stage_best_gtmask_signloss_v8_enhance_0cons_optv2.pth"
output_dir = "/mnt/d/nuoqi/S2R-HDR/data/test_vivo/rf3stage_v8_output"
save_output = False

norm_type = "group"
gn_groups = 8

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


def compute_sobel_grad(x, eps=1e-6, normalize=False):
    sobel_x = torch.tensor([[[[-1.0, 0.0, 1.0],
                              [-2.0, 0.0, 2.0],
                              [-1.0, 0.0, 1.0]]]], device=x.device, dtype=x.dtype)
    sobel_y = torch.tensor([[[[-1.0, -2.0, -1.0],
                              [0.0, 0.0, 0.0],
                              [1.0, 2.0, 1.0]]]], device=x.device, dtype=x.dtype)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + eps)
    if normalize:
        g = g / (g.amax(dim=(2, 3), keepdim=True) + eps)
    return g, gx, gy


def batched_otsu_threshold(g, num_bins=256, eps=1e-6, min_delta=1e-6):
    assert g.dim() == 4 and g.shape[1] == 1, "g must be (B,1,H,W)"
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")

    batch_size = g.shape[0]
    flat = g.flatten(1)

    g_min = flat.min(dim=1).values
    g_max = flat.max(dim=1).values
    g_span = g_max - g_min
    span = g_span.clamp_min(eps)

    norm = ((flat - g_min[:, None]) / span[:, None]).clamp(0.0, 1.0)
    bin_idx = (norm * (num_bins - 1)).long()

    offsets = (torch.arange(batch_size, device=g.device) * num_bins)[:, None]
    hist_idx = (bin_idx + offsets).reshape(-1)
    hist = torch.bincount(hist_idx, minlength=batch_size * num_bins).reshape(batch_size, num_bins).to(g.dtype)

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

    valid_span = g_span > min_delta
    high = g_max - min_delta
    thresh = torch.where(valid_span, torch.minimum(thresh, high), g_max)
    return thresh


def gaussian_kernel1d(kernel_size, sigma, device, dtype):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel


def gaussian_blur(x, kernel_size=5, sigma=1.0):
    if x.dim() != 4:
        raise ValueError("x must be 4D (B,C,H,W)")

    _, channels, _, _ = x.shape
    k1 = gaussian_kernel1d(kernel_size, sigma, x.device, x.dtype)
    kx = k1.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    ky = k1.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    x = F.conv2d(x, kx, padding=(0, kernel_size // 2), groups=channels)
    x = F.conv2d(x, ky, padding=(kernel_size // 2, 0), groups=channels)
    return x


def build_edge_mask_from_grad(g, thresh_mode="otsu", thresh=0.9, dilate=1, eps=1e-6):
    if thresh_mode == "fixed":
        threshold = torch.as_tensor(thresh, device=g.device, dtype=g.dtype)
        edge_mask = (g > threshold).float()
    elif thresh_mode == "percentile":
        q = torch.quantile(g.flatten(1), torch.as_tensor(thresh, device=g.device), dim=1)
        edge_mask = (g > q.view(-1, 1, 1, 1)).float()
    elif thresh_mode == "otsu":
        threshold = batched_otsu_threshold(g, num_bins=256, eps=eps, min_delta=eps)
        edge_mask = (g > threshold.view(-1, 1, 1, 1)).float()

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
        kernel = dilate * 2 + 1
        edge_mask = F.max_pool2d(edge_mask, kernel_size=kernel, stride=1, padding=dilate)

    return edge_mask


def per_channel_loss(pred, target, loss_type="l1"):
    if loss_type == "l1":
        return (pred - target).abs()
    if loss_type == "l2":
        diff = pred - target
        return diff * diff
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


def build_sign_supervision_mask(target, thresh_low=3e-8):
    return (target.abs() > thresh_low).float()


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
    signed_margin = sign_target * sign_logits
    loss_map = F.softplus(-signed_margin)
    if weight is not None:
        loss_map = loss_map * weight
    return masked_reduce(loss_map, mask=mask, eps=eps, reduction="mean")


def edge_guided_loss(
    pred,
    target,
    thresh_mode="otsu",
    thresh=0.9,
    dilate=1,
    loss_type="l1",
    eps=1e-6,
    normalize_grad=True,
    detach_mask=True,
    lowfreq_kernel=5,
    lowfreq_sigma=1.0,
    w_dir=0.3,
    w_lp=2.0,
    smooth_alpha=1.0,
    smooth_weight_mode="smooth_over_edge",
    precomputed_edge_mask=None,
    precomputed_smooth_mask=None,
    precomputed_gt_lp=None,
):
    assert pred.shape == target.shape

    if precomputed_edge_mask is None:
        g, _, _ = compute_sobel_grad(target, eps=eps, normalize=normalize_grad)
        edge_mask = build_edge_mask_from_grad(g, thresh_mode=thresh_mode, thresh=thresh, dilate=dilate, eps=eps)
    else:
        edge_mask = precomputed_edge_mask

    if detach_mask:
        edge_mask = edge_mask.detach()

    if precomputed_smooth_mask is None:
        smooth_mask = (1.0 - edge_mask).clamp(0.0, 1.0)
    else:
        smooth_mask = precomputed_smooth_mask

    per_pix = per_channel_loss(pred, target, loss_type=loss_type)
    if per_pix.dim() == 4 and per_pix.shape[1] > 1:
        per_pix = per_pix.mean(dim=1, keepdim=True)
    edge_pix = masked_reduce(per_pix, edge_mask, eps=eps, reduction="mean")

    _, gpx, gpy = compute_sobel_grad(pred, eps=eps, normalize=normalize_grad)
    _, gtx, gty = compute_sobel_grad(target, eps=eps, normalize=normalize_grad)
    grad_diff = (gpx - gtx).abs() + (gpy - gty).abs()
    edge_grad = masked_reduce(grad_diff, edge_mask, eps=eps, reduction="mean")

    pred_mag = torch.sqrt(gpx * gpx + gpy * gpy + eps)
    gt_mag = torch.sqrt(gtx * gtx + gty * gty + eps)
    dir_px = gpx / pred_mag
    dir_py = gpy / pred_mag
    dir_tx = gtx / gt_mag
    dir_ty = gty / gt_mag
    dir_diff = 1.0 - (dir_px * dir_tx + dir_py * dir_ty)
    edge_dir = masked_reduce(dir_diff, edge_mask, eps=eps, reduction="mean")

    pred_lp = gaussian_blur(pred, kernel_size=lowfreq_kernel, sigma=lowfreq_sigma)
    if precomputed_gt_lp is None:
        gt_lp = gaussian_blur(target, kernel_size=lowfreq_kernel, sigma=lowfreq_sigma)
    else:
        gt_lp = precomputed_gt_lp
    lp_pix = per_channel_loss(pred_lp, gt_lp, loss_type=loss_type)
    if lp_pix.dim() == 4 and lp_pix.shape[1] > 1:
        lp_pix = lp_pix.mean(dim=1, keepdim=True)
    smooth_lp = masked_reduce(lp_pix, smooth_mask, eps=eps, reduction="mean")

    edge_area = edge_mask.flatten(1).sum(dim=1)
    smooth_area = smooth_mask.flatten(1).sum(dim=1)
    if smooth_weight_mode == "edge_over_smooth":
        area_ratio = edge_area / (smooth_area + eps)
    elif smooth_weight_mode == "smooth_over_edge":
        area_ratio = smooth_area / (edge_area + eps)
    else:
        raise ValueError("smooth_weight_mode must be 'edge_over_smooth' or 'smooth_over_edge'")
    smooth_w = (area_ratio.clamp_min(eps) ** smooth_alpha).mean()
    smooth_w = smooth_w.clamp(min=0.1, max=10.0)

    ratio_edge = (edge_grad / (edge_pix + eps)).detach().clamp_min(eps)
    ratio_lp = (smooth_lp / (edge_pix + eps)).detach().clamp_min(eps)
    total = edge_pix + edge_grad / ratio_edge + w_dir * edge_dir + (w_lp * smooth_w) * (smooth_lp / ratio_lp)
    return total


def cascade_loss(
    outputs,
    x_coarse0,
    gt,
    thresh_mode="otsu",
    thresh=0.6,
    dilate=0,
    loss_type="l1",
    normalize_grad=True,
    precomputed_edge_mask=None,
    precomputed_smooth_mask=None,
    precomputed_gt_lp=None,
    sign_ignore_thresh=3e-8,
    sign_small_tau=2e-2,
    sign_small_gain=2.0,
    sign_smooth_gain=1.0,
    stage1_sign_weight=6.0,
    stage1_edge_weight=2.0,
    stage2_weight=0.7,
    stage3_weight=1.0,
    final_weight=0.5,
):
    delta_gt = gt - x_coarse0

    l1 = edge_guided_loss(
        outputs["delta1"],
        delta_gt,
        thresh_mode=thresh_mode,
        thresh=thresh,
        dilate=dilate,
        loss_type=loss_type,
        normalize_grad=normalize_grad,
        precomputed_edge_mask=precomputed_edge_mask,
        precomputed_smooth_mask=precomputed_smooth_mask,
        precomputed_gt_lp=precomputed_gt_lp,
    )
    l2 = edge_guided_loss(
        outputs["delta2"],
        delta_gt,
        thresh_mode=thresh_mode,
        thresh=thresh,
        dilate=dilate,
        loss_type=loss_type,
        normalize_grad=normalize_grad,
        precomputed_edge_mask=precomputed_edge_mask,
        precomputed_smooth_mask=precomputed_smooth_mask,
        precomputed_gt_lp=precomputed_gt_lp,
    )
    l3 = edge_guided_loss(
        outputs["delta3"],
        delta_gt,
        thresh_mode=thresh_mode,
        thresh=thresh,
        dilate=dilate,
        loss_type=loss_type,
        normalize_grad=normalize_grad,
        precomputed_edge_mask=precomputed_edge_mask,
        precomputed_smooth_mask=precomputed_smooth_mask,
        precomputed_gt_lp=precomputed_gt_lp,
    )

    sign_mask = build_sign_supervision_mask(delta_gt, thresh_low=sign_ignore_thresh)
    sign_weight = build_sign_weight_map(
        delta_gt,
        smooth_mask=precomputed_smooth_mask,
        small_focus_tau=sign_small_tau,
        small_focus_gain=sign_small_gain,
        smooth_focus_gain=sign_smooth_gain,
    )
    stage1_sign = sign_consistency_loss(
        outputs["stage1_sign_logits"],
        delta_gt,
        mask=sign_mask,
        weight=sign_weight,
    )
    final_l1 = F.l1_loss(outputs["final"], gt)

    total = (
        stage1_edge_weight * l1
        + stage2_weight * l2
        + stage3_weight * l3
        + stage1_sign_weight * stage1_sign
        + final_weight * final_l1
    )

    aux = {
        "stage1_edge": l1,
        "stage2_edge": l2,
        "stage3_edge": l3,
        "stage1_sign": stage1_sign,
        "final_l1": final_l1,
    }
    return total, aux


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
            nn.Conv2d(base_channels, 2, kernel_size=1, bias=True),
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

        in_channels = 3 + (1 if use_delta_as_input else 0)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            make_norm(16, norm_type=norm_type, gn_groups=gn_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, base_channels, 3, padding=1, bias=False),
            make_norm(base_channels, norm_type=norm_type, gn_groups=gn_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            make_norm(base_channels, norm_type=norm_type, gn_groups=gn_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 2, 1, bias=True),
        )

    def forward(self, x_raw, x_coarse_k, delta_prev=None):
        grad, _, _ = compute_sobel_grad(x_coarse_k, normalize=True)
        if self.use_delta_as_input:
            if delta_prev is None:
                raise ValueError("delta_prev is required when use_delta_as_input=True")
            inp = torch.cat([x_raw, x_coarse_k, grad, delta_prev], dim=1)
        else:
            inp = torch.cat([x_raw, x_coarse_k, grad], dim=1)

        ab = self.net(inp)
        a_raw = ab[:, 0:1]
        b_raw = ab[:, 1:2]
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
        self.stage2 = ABAdjustNet(
            base_channels=base_channels,
            a_scale=a_scale,
            b_scale=b_scale,
            use_delta_as_input=use_delta_as_input,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )
        self.stage3 = ABAdjustNet(
            base_channels=base_channels,
            a_scale=a_scale,
            b_scale=b_scale,
            use_delta_as_input=use_delta_as_input,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )

    def forward(self, x_raw, x_coarse0):
        stage1_out = self.stage1(x_raw, x_coarse0)
        delta1 = stage1_out["delta"]
        x_coarse1 = x_coarse0 + delta1

        a2, b2 = self.stage2(x_raw, x_coarse1, delta1)
        delta2 = a2 * delta1 + b2
        x_coarse2 = x_coarse0 + delta2

        a3, b3 = self.stage3(x_raw, x_coarse2, delta2)
        delta3 = a3 * delta2 + b3
        x_final = x_coarse0 + delta3

        return {
            "delta1": delta1,
            "coarse1": x_coarse1,
            "delta2": delta2,
            "coarse2": x_coarse2,
            "delta3": delta3,
            "final": x_final,
            "stage1_sign_logits": stage1_out["sign_logits"],
            "stage1_sign_map": stage1_out["sign_map"],
            "stage1_mag_map": stage1_out["mag_map"],
            "a2": a2,
            "b2": b2,
            "a3": a3,
            "b3": b3,
        }


def save_float32_tiff(img, save_path):
    img = img.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    imageio.imwrite(save_path, img.astype(np.float32))


def crop_to_even(tensor):
    _, _, height, width = tensor.shape
    new_height = height if height % 2 == 0 else height - 1
    new_width = width if width % 2 == 0 else width - 1
    return tensor[:, :, :new_height, :new_width]


class GuidedFilterDataset(Dataset):
    def __init__(self, origin_input_dir, input_dir, label_dir):
        self.origin_input_files = sorted([os.path.join(origin_input_dir, f) for f in os.listdir(origin_input_dir)])
        self.input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        print(f"INPUT: {len(self.input_files)}, Label: {len(self.label_files)}")

    def __len__(self):
        return min(len(self.input_files), len(self.label_files))

    def __getitem__(self, idx):
        input_path = self.input_files[idx]
        origin_input = cv2.imread(self.origin_input_files[idx], cv2.IMREAD_UNCHANGED).astype("float32")
        coarse_input = cv2.imread(self.input_files[idx], cv2.IMREAD_UNCHANGED).astype("float32")
        label = cv2.imread(self.label_files[idx], cv2.IMREAD_UNCHANGED).astype("float32")
        input_height, input_width = coarse_input.shape
        filename = os.path.splitext(os.path.basename(input_path))[0]

        return {
            "origin_input_hr": torch.from_numpy(origin_input).unsqueeze(0).unsqueeze(0),
            "input_hr": torch.from_numpy(coarse_input).unsqueeze(0).unsqueeze(0),
            "target_img": torch.from_numpy(label).unsqueeze(0).unsqueeze(0),
            "filename": filename,
            "input_height_hr": input_height,
            "input_width_hr": input_width,
        }


def var_custom_collate(batch):
    min_h_hr = min(item["input_height_hr"] for item in batch)
    min_w_hr = min(item["input_width_hr"] for item in batch)

    batch_input = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_input_origin = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_target_images = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    filenames = []

    for idx, item in enumerate(batch):
        batch_input_origin[idx] = item["origin_input_hr"][:, :, :min_h_hr, :min_w_hr]
        batch_input[idx] = item["input_hr"][:, :, :min_h_hr, :min_w_hr]
        batch_target_images[idx] = item["target_img"][:, :, :min_h_hr, :min_w_hr]
        filenames.append(item["filename"])

    return batch_input_origin, batch_input, batch_target_images, filenames


def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1 / mse)
    return psnr, mse


def build_model():
    model = CascadeRefine3Stage(
        base_channels=32,
        delta_scale=0.40,
        a_scale=0.5,
        b_scale=0.4,
        use_delta_as_input=True,
        norm_type=norm_type,
        gn_groups=gn_groups,
    )
    model = nn.DataParallel(model).to(device)
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate():
    if save_output:
        os.makedirs(output_dir, exist_ok=True)

    model = build_model()
    test_dataset = GuidedFilterDataset(
        origin_input_dir=origin_input_test,
        input_dir=input_folder_test,
        label_dir=input_folder_test_label,
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

    test_mse = 0.0
    test_psnr = 0.0
    test_loss = 0.0
    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for test_input_origin, test_input, test_label, filenames in tqdm(test_loader, desc="Testing"):
            test_input_origin = crop_to_even(test_input_origin)
            test_input = crop_to_even(test_input)
            test_label = crop_to_even(test_label)

            test_input_origin = test_input_origin.to(device)
            test_input = test_input.to(device)
            test_label = test_label.to(device)

            test_input_origin = test_input_origin - 0.5
            test_input = test_input - 0.5
            test_label = test_label - 0.5

            start_time = time.time()
            outputs = model(test_input_origin, test_input)
            end_time = time.time()

            delta_gt = test_label - test_input
            grad, _, _ = compute_sobel_grad(test_label, eps=1e-6, normalize=True)
            test_edge_mask = build_edge_mask_from_grad(
                grad, thresh_mode="otsu", thresh=0.9, dilate=0, eps=1e-6
            )
            test_smooth_mask = (1.0 - test_edge_mask).clamp(0.0, 1.0)
            test_gt_lp = gaussian_blur(
                delta_gt,
                kernel_size=DEFAULT_LOWFREQ_KERNEL,
                sigma=DEFAULT_LOWFREQ_SIGMA,
            )

            loss, _ = cascade_loss(
                outputs,
                test_input,
                test_label,
                thresh_mode="otsu",
                dilate=0,
                loss_type="l1",
                normalize_grad=True,
                precomputed_edge_mask=test_edge_mask,
                precomputed_smooth_mask=test_smooth_mask,
                precomputed_gt_lp=test_gt_lp,
            )

            test_output = torch.clamp(outputs["final"] + 0.5, 0, 1)
            test_label_vis = test_label + 0.5

            total_time += end_time - start_time
            total_images += test_input.size(0)

            if save_output:
                for idx, filename in enumerate(filenames):
                    save_float32_tiff(test_output[idx], os.path.join(output_dir, f"{filename}.tiff"))

            psnr, mse = calc_psnr(test_output.detach(), test_label_vis)
            test_mse += mse
            test_psnr += psnr
            test_loss += loss.item()

    avg_test_mse = test_mse / len(test_loader)
    avg_test_psnr = test_psnr / len(test_loader)
    avg_test_loss = test_loss / len(test_loader)
    avg_infer_time = total_time / max(total_images, 1)

    print(
        f"Test Finished - Average PSNR: {avg_test_psnr:.2f} dB, "
        f"Average MSE: {avg_test_mse:.6f}, Average Loss: {avg_test_loss:.6f}"
    )
    print(f"Average inference time per image: {avg_infer_time * 1000:.2f} ms")


if __name__ == "__main__":
    evaluate()
