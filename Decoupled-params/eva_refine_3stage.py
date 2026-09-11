import os
import re
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F
import imageio

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

class ParamConditioning(nn.Module):
    def __init__(self, channels, param_dim=2, hidden_dim=16):
        super().__init__()
        self.fc1_weight = nn.Linear(param_dim, hidden_dim)
        self.fc1_bias = nn.Linear(param_dim, hidden_dim)
        self.fc2_weight = nn.Linear(hidden_dim, channels)
        self.fc2_bias = nn.Linear(hidden_dim, channels)
        self.param_relu = nn.ReLU(inplace=True)

    def encode(self, params):
        norm_weight = self.fc2_weight(self.param_relu(self.fc1_weight(params)))
        norm_bias = self.fc2_bias(self.param_relu(self.fc1_bias(params)))
        return norm_weight, norm_bias

def apply_param_injection(feat, norm_weight, norm_bias):
    batch_outputs = []
    for n in range(feat.size(0)):
        feat_n = F.instance_norm(
            feat.narrow(0, n, 1),
            weight=norm_weight[n],
            bias=norm_bias[n],
        )
        batch_outputs.append(feat_n)
    return torch.cat(batch_outputs, dim=0)

class Stage1DeltaNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, delta_scale=0.4):
        super().__init__()
        self.delta_scale = delta_scale
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.inject_conv = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False)
        self.param_condition = ParamConditioning(base_channels)
        self.act = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1, bias=False)

    def forward(self, x_raw, x_coarse, params):
        g, _, _ = compute_sobel_grad(x_coarse, normalize=True)
        inp = torch.cat([x_raw, x_coarse, g], dim=1)
        feat = self.head(inp)
        feat = self.body(feat)
        feat = self.inject_conv(feat)
        norm_weight, norm_bias = self.param_condition.encode(params)
        feat = self.act(apply_param_injection(feat, norm_weight, norm_bias))
        delta = self.delta_scale * torch.tanh(self.out_conv(feat))
        return delta

class ABAdjustNet(nn.Module):
    def __init__(self, base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True):
        super().__init__()
        self.a_scale = a_scale
        self.b_scale = b_scale
        self.use_delta_as_input = use_delta_as_input

        in_ch = 3 + (1 if use_delta_as_input else 0)

        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.inject_conv = nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False)
        self.param_condition = ParamConditioning(base_channels)
        self.act = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(base_channels, 2, 1, bias=True)

    def forward(self, x_raw, x_coarse_k, params, delta_prev=None):
        g, _, _ = compute_sobel_grad(x_coarse_k, normalize=True)

        if self.use_delta_as_input:
            assert delta_prev is not None
            inp = torch.cat([x_raw, x_coarse_k, g, delta_prev], dim=1)
        else:
            inp = torch.cat([x_raw, x_coarse_k, g], dim=1)

        feat = self.head(inp)
        feat = self.body(feat)
        feat = self.inject_conv(feat)
        norm_weight, norm_bias = self.param_condition.encode(params)
        feat = self.act(apply_param_injection(feat, norm_weight, norm_bias))
        ab = self.out_conv(feat)
        a_raw = ab[:, 0:1]
        b_raw = ab[:, 1:2]

        a = 1.0 + self.a_scale * torch.tanh(a_raw)
        b = self.b_scale * torch.tanh(b_raw)
        return a, b

class CascadeRefine3Stage(nn.Module):
    def __init__(self, base_channels=32, a_scale=0.5, b_scale=0.4, delta_scale=0.4, use_delta_as_input=True):
        super().__init__()
        self.stage1 = Stage1DeltaNet(base_channels=base_channels, delta_scale=delta_scale)
        self.stage2 = ABAdjustNet(base_channels=base_channels, a_scale=a_scale, b_scale=b_scale,
                                 use_delta_as_input=use_delta_as_input)
        self.stage3 = ABAdjustNet(base_channels=base_channels, a_scale=a_scale, b_scale=b_scale,
                                 use_delta_as_input=use_delta_as_input)

    def forward(self, x_raw, x_coarse0, params):
        delta1 = self.stage1(x_raw, x_coarse0, params)
        x_coarse1 = x_coarse0 + delta1

        a2, b2 = self.stage2(x_raw, x_coarse1, params, delta1)
        delta2 = a2 * delta1 + b2
        x_coarse2 = x_coarse0 + delta2

        a3, b3 = self.stage3(x_raw, x_coarse2, params, delta2)
        delta3 = a3 * delta2 + b3
        x_final = x_coarse0 + delta3

        return {
            "delta1": delta1, "coarse1": x_coarse1,
            "delta2": delta2, "coarse2": x_coarse2,
            "delta3": delta3, "final":  x_final,
            "a2": a2, "b2": b2,
            "a3": a3, "b3": b3,
        }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

origin_input_test = os.environ.get(
    "RF3_ORIGIN_INPUT_TEST",
    '/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/testinput'
)
input_folder_test = os.environ.get(
    "RF3_INPUT_FOLDER_TEST",
    '/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/test_refine_input'
)
input_folder_test_label = os.environ.get(
    "RF3_INPUT_FOLDER_TEST_LABEL",
    '/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/testlabel'
)

model_path = os.environ.get(
    "RF3_MODEL_PATH",
    "/mnt/d/nuoqi/halo-free/Decoupled-params/model/rf_3stage_best_otsu_v3_falsenorm_param.pth"
)

output_dir = os.environ.get(
    "RF3_OUTPUT_DIR",
    "/mnt/d/nuoqi/S2R-HDR/data/mixed-lavel-singlehdr-param/test_rf_3stage_output"
)
os.makedirs(output_dir, exist_ok=True)

model = CascadeRefine3Stage(base_channels=32, a_scale=0.5, b_scale=0.4, delta_scale=0.4, use_delta_as_input=True)
model = nn.DataParallel(model).to(device)

checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
model.load_state_dict(state_dict)

model.eval()

def save_float32_tiff(img, save_path):
    """
    直接保存原始 img 为 float32 tiff，不做归一化。
    保留负值和原始动态范围。
    """
    img = img.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32)
    imageio.imwrite(save_path, img)

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

def compute_sobel_grad(x, eps=1e-6, normalize=False):
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
    return g, gx, gy

def batched_otsu_threshold(g, num_bins=256, eps=1e-6, min_delta=1e-6):
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
    _, C, _, _ = x.shape
    k1 = gaussian_kernel1d(kernel_size, sigma, x.device, x.dtype)
    kx = k1.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    ky = k1.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    x = F.conv2d(x, kx, padding=(0, kernel_size // 2), groups=C)
    x = F.conv2d(x, ky, padding=(kernel_size // 2, 0), groups=C)
    return x

DEFAULT_LOWFREQ_KERNEL = 5
DEFAULT_LOWFREQ_SIGMA = 1.0

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

def per_channel_loss(pred, target, loss_type='l1'):
    if loss_type == 'l1':
        return (pred - target).abs()
    if loss_type == 'l2':
        d = pred - target
        return d * d
    if loss_type == 'huber':
        return F.smooth_l1_loss(pred, target, reduction='none')
    raise ValueError("loss_type must be 'l1' | 'l2' | 'huber'")

def masked_reduce(per, mask=None, eps=1e-8, reduction='mean'):
    if mask is not None:
        per = per * mask
        if reduction == 'mean':
            denom = mask.sum().clamp_min(eps)
            return per.sum() / denom
        if reduction == 'sum':
            return per.sum()
        return per
    if reduction == 'mean':
        return per.mean()
    if reduction == 'sum':
        return per.sum()
    return per

def edge_guided_loss(
    pred, target,
    thresh_mode='fixed',
    thresh=0.6,
    dilate=1,
    loss_type='l1',
    eps=1e-6,
    normalize_grad=True,
    detach_mask=True,
    lowfreq_kernel=5,
    lowfreq_sigma=1.0,
    w_dir=0.3,
    w_lp=2.0,
    smooth_alpha=1.0,
    smooth_weight_mode='smooth_over_edge',
    precomputed_edge_mask=None,
    precomputed_smooth_mask=None,
    precomputed_gt_lp=None,
):
    assert pred.shape == target.shape

    if precomputed_edge_mask is None:
        g, _, _ = compute_sobel_grad(target, eps=eps, normalize=normalize_grad)
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

    per_pix = per_channel_loss(pred, target, loss_type=loss_type)
    if per_pix.dim() == 4 and per_pix.shape[1] > 1:
        per_pix = per_pix.mean(dim=1, keepdim=True)

    edge_pix = masked_reduce(per_pix, edge_mask, eps=eps, reduction='mean')

    _, gpx, gpy = compute_sobel_grad(pred,   eps=eps, normalize=normalize_grad)
    _, gtx, gty = compute_sobel_grad(target, eps=eps, normalize=normalize_grad)
    grad_diff = (gpx - gtx).abs() + (gpy - gty).abs()
    edge_grad = masked_reduce(grad_diff, edge_mask, eps=eps, reduction='mean')

    pred_mag = torch.sqrt(gpx * gpx + gpy * gpy + eps)
    gt_mag = torch.sqrt(gtx * gtx + gty * gty + eps)
    dir_px = gpx / pred_mag
    dir_py = gpy / pred_mag
    dir_tx = gtx / gt_mag
    dir_ty = gty / gt_mag
    dir_diff = 1.0 - (dir_px * dir_tx + dir_py * dir_ty)
    edge_dir = masked_reduce(dir_diff, edge_mask, eps=eps, reduction='mean')

    pred_lp = gaussian_blur(pred, kernel_size=lowfreq_kernel, sigma=lowfreq_sigma)
    if precomputed_gt_lp is None:
        gt_lp = gaussian_blur(target, kernel_size=lowfreq_kernel, sigma=lowfreq_sigma)
    else:
        gt_lp = precomputed_gt_lp
    lp_pix = per_channel_loss(pred_lp, gt_lp, loss_type=loss_type)
    if lp_pix.dim() == 4 and lp_pix.shape[1] > 1:
        lp_pix = lp_pix.mean(dim=1, keepdim=True)
    smooth_lp = masked_reduce(lp_pix, smooth_mask, eps=eps, reduction='mean')

    edge_area = edge_mask.flatten(1).sum(dim=1)
    smooth_area = smooth_mask.flatten(1).sum(dim=1)
    if smooth_weight_mode == 'edge_over_smooth':
        area_ratio = edge_area / (smooth_area + eps)
    elif smooth_weight_mode == 'smooth_over_edge':
        area_ratio = smooth_area / (edge_area + eps)
    else:
        raise ValueError("smooth_weight_mode must be 'edge_over_smooth' or 'smooth_over_edge'")
    smooth_w = (area_ratio.clamp_min(eps) ** smooth_alpha).mean()
    smooth_w = smooth_w.clamp(min=0.1, max=10.0)

    ratio_edge = (edge_grad / (edge_pix + eps)).detach().clamp_min(eps)
    ratio_lp = (smooth_lp / (edge_pix + eps)).detach().clamp_min(eps)
    total = edge_pix \
        + edge_grad  / ratio_edge \
        + w_dir * edge_dir \
        + (w_lp * smooth_w) * (smooth_lp / ratio_lp)

    return total

def cascade_loss(outputs, x_coarse0, gt,
                 thresh_mode='otsu', thresh=0.6, dilate=0,
                 loss_type='l1', normalize_grad=True,
                 precomputed_edge_mask=None,
                 precomputed_smooth_mask=None,
                 precomputed_gt_lp=None):
    delta_gt = gt - x_coarse0

    l1 = edge_guided_loss(outputs["delta1"],  delta_gt,
                          thresh_mode=thresh_mode, thresh=thresh, dilate=dilate,
                          loss_type=loss_type, normalize_grad=normalize_grad,
                          precomputed_edge_mask=precomputed_edge_mask,
                          precomputed_smooth_mask=precomputed_smooth_mask,
                          precomputed_gt_lp=precomputed_gt_lp)
    l2 = edge_guided_loss(outputs["delta2"], delta_gt,
                          thresh_mode=thresh_mode, thresh=thresh, dilate=dilate,
                          loss_type=loss_type, normalize_grad=normalize_grad,
                          precomputed_edge_mask=precomputed_edge_mask,
                          precomputed_smooth_mask=precomputed_smooth_mask,
                          precomputed_gt_lp=precomputed_gt_lp)
    l3 = edge_guided_loss(outputs["delta3"], delta_gt,
                          thresh_mode=thresh_mode, thresh=thresh, dilate=dilate,
                          loss_type=loss_type, normalize_grad=normalize_grad,
                          precomputed_edge_mask=precomputed_edge_mask,
                          precomputed_smooth_mask=precomputed_smooth_mask,
                          precomputed_gt_lp=precomputed_gt_lp)

    return 0.33*l1 + 0.33*l2 + 0.33*l3, (l1, l2, l3)

def parse_wls_params(path):
    name = os.path.basename(path)
    match = re.search(r'-l([0-9eE+\-.]+)-a([0-9eE+\-.]+)-h', name)
    if match is None:
        raise ValueError(f"Cannot parse lambda/alpha from filename: {name}")
    return float(match.group(1)), float(match.group(2))

def _strip_all_extensions(filename):
    root = filename
    while True:
        root_next, ext = os.path.splitext(root)
        if not ext:
            return root
        root = root_next

def resolve_origin_path(input_path, origin_index):
    input_name = os.path.basename(input_path)
    input_stem = _strip_all_extensions(input_name)

    exact = origin_index.get(input_name)
    if exact is not None:
        return exact

    if input_stem in origin_index:
        return origin_index[input_stem]

    prefix = input_stem.split('-wls-')[0]
    candidates = [
        prefix,
        f"{prefix}-log",
        f"{prefix}_log",
    ]
    for key in candidates:
        matched = origin_index.get(key)
        if matched is not None:
            return matched

    raise FileNotFoundError(f"Cannot find matching origin input for: {input_name}")

class GuidedFilterDataset(Dataset):
    def __init__(self, origin_input_dir, input_dir, label_dir):
        self.origin_input_files = sorted([
            os.path.join(origin_input_dir, f) for f in os.listdir(origin_input_dir)
        ])
        self.input_files = sorted([
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
        ])
        self.label_files = sorted([
            os.path.join(label_dir, f) for f in os.listdir(label_dir)
        ])

        self.origin_index = {}
        for path in self.origin_input_files:
            name = os.path.basename(path)
            self.origin_index[name] = path
            self.origin_index[_strip_all_extensions(name)] = path

        self.label_index = {
            os.path.basename(path): path for path in self.label_files
        }
        self.samples = []
        for input_path in self.input_files:
            name = os.path.basename(input_path)
            label_path = self.label_index.get(name)
            if label_path is None:
                raise FileNotFoundError(f"Cannot find matching label for: {name}")
            origin_path = resolve_origin_path(input_path, self.origin_index)
            self.samples.append((origin_path, input_path, label_path))

        print(f"INPUT: {len(self.input_files)}, Label: {len(self.label_files)}, Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        origin_path, input_path, label_path = self.samples[idx]
        origin_input = cv2.imread(origin_path, cv2.IMREAD_UNCHANGED).astype('float32')
        input = cv2.imread(input_path, cv2.IMREAD_UNCHANGED).astype('float32')
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED).astype('float32')
        lambda_val, alpha_val = parse_wls_params(input_path)
        input_hr_h, input_hr_w = input.shape

        filename = os.path.splitext(os.path.basename(input_path))[0]

        return {
            'origin_input_hr': torch.from_numpy(origin_input).unsqueeze(0).unsqueeze(0),
            'input_hr': torch.from_numpy(input).unsqueeze(0).unsqueeze(0),
            'target_img': torch.from_numpy(label).unsqueeze(0).unsqueeze(0),
            'input_params': torch.tensor([lambda_val, alpha_val], dtype=torch.float32),
            'filename': filename,
            'input_height_hr': input_hr_h,
            'input_width_hr': input_hr_w,
        }

def var_custom_collate(batch):
    min_h_hr = min([item['input_height_hr'] for item in batch])
    min_w_hr = min([item['input_width_hr'] for item in batch])

    batch_input = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_input_origin   = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_target_images = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_params = torch.zeros(len(batch), 2)
    filenames = []

    for idx, item in enumerate(batch):
        input_origin_crop = item['origin_input_hr'][:, :, :min_h_hr, :min_w_hr]
        input_crop   = item['input_hr'][:, :, :min_h_hr, :min_w_hr]
        label_crop  = item['target_img'][:, :, :min_h_hr, :min_w_hr]

        batch_input_origin[idx] = input_origin_crop
        batch_input[idx] = input_crop
        batch_target_images[idx] = label_crop
        batch_params[idx] = item['input_params']
        filenames.append(item['filename'])

    return batch_input_origin, batch_input, batch_target_images, batch_params, filenames

test_dataset = GuidedFilterDataset(
    origin_input_dir = origin_input_test,
    input_dir = input_folder_test,
    label_dir = input_folder_test_label
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                          collate_fn=var_custom_collate, pin_memory=True, drop_last=False)

def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1 / mse)
    return psnr, mse

def main():
    test_mse = 0
    test_psnr = 0
    test_loss = 0
    total_time = 0
    total_images = 0

    with torch.no_grad():
        for test_input_origin, test_input, test_label, test_params, filenames in tqdm(test_loader, desc="Testing"):
            test_input_origin = crop_to_even(test_input_origin)
            test_input = crop_to_even(test_input)
            test_label = crop_to_even(test_label)

            test_input_origin = test_input_origin.to(device)
            test_input = test_input.to(device)
            test_label = test_label.to(device)
            test_params = test_params.to(device)

            test_input_origin = test_input_origin - 0.5
            test_input = test_input - 0.5
            test_label = test_label - 0.5

            start_time = time.time()
            outputs = model(test_input_origin, test_input, test_params)
            end_time = time.time()

            delta_gt = test_label - test_input
            g, _, _ = compute_sobel_grad(delta_gt, eps=1e-6, normalize=True)
            test_edge_mask = build_edge_mask_from_grad(
                g, thresh_mode='otsu', thresh=0.9, dilate=0, eps=1e-6
            )
            test_smooth_mask = (1.0 - test_edge_mask).clamp(0.0, 1.0)
            test_gt_lp = gaussian_blur(
                delta_gt, kernel_size=DEFAULT_LOWFREQ_KERNEL, sigma=DEFAULT_LOWFREQ_SIGMA
            )

            loss, _ = cascade_loss(
                outputs, test_input, test_label,
                thresh_mode='otsu',
                dilate=0,
                loss_type='l1',
                normalize_grad=True,
                precomputed_edge_mask=test_edge_mask,
                precomputed_smooth_mask=test_smooth_mask,
                precomputed_gt_lp=test_gt_lp
            )

            test_output = outputs["final"] + 0.5
            test_output = torch.clamp(test_output, 0, 1)
            test_label = test_label + 0.5

            batch_time = end_time - start_time
            total_time += batch_time
            total_images += test_input.size(0)

            for i, fname in enumerate(filenames):
                save_float32_tiff(test_output[i], f"{output_dir}/{fname}.tiff")

            psnr, mse = calc_psnr(test_output.detach(), test_label)
            test_mse += mse
            test_psnr += psnr
            test_loss += loss.item()

    avg_test_mse = test_mse / len(test_loader)
    avg_test_psnr = test_psnr / len(test_loader)
    avg_test_loss = test_loss / len(test_loader)
    avg_infer_time = total_time / total_images

    print(f"Test Finished - Average PSNR: {avg_test_psnr:.2f} dB, Average MSE: {avg_test_mse:.6f}, Average Loss: {avg_test_loss:.6f}")
    print(f"Average inference time per image: {avg_infer_time * 1000:.2f} ms")

if __name__ == "__main__":
    main()
