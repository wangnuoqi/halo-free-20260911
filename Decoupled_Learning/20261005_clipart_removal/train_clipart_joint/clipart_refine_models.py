import torch
import torch.nn as nn
import torch.nn.functional as F

class Stage1DeltaNet(nn.Module):
    def __init__(self, channels=1, base_channels=32):
        in_channels = 3 * channels
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

            nn.Conv2d(base_channels, 2 * channels, kernel_size=1, bias=True)  # raw delta, raw edge gate
        )

    def forward(self, x_raw, x_coarse):
        g, _, _ = compute_sobel_grad(x_coarse, normalize=True)
        inp = torch.cat([x_raw, x_coarse, g], dim=1)
        raw = self.net(inp)
        delta_raw = raw[:, :raw.shape[1] // 2]
        gate = torch.sigmoid(raw[:, raw.shape[1] // 2:])
        return delta_raw * gate, gate

class ABAdjustNet(nn.Module):
    def __init__(self, base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True, channels=1):
        super().__init__()
        self.a_scale = a_scale
        self.b_scale = b_scale
        self.use_delta_as_input = use_delta_as_input

        in_ch = channels * (3 + (1 if use_delta_as_input else 0))  # raw + coarse_k + grad + (delta_prev可选)

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

            nn.Conv2d(base_channels, 3 * channels, 1, bias=True)  # (a_raw, b_raw, gate_raw) each pixel
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
        c = ab.shape[1] // 3
        a_raw = ab[:, :c]
        b_raw = ab[:, c:2*c]
        gate = torch.sigmoid(ab[:, 2*c:])

        # per-pixel a,b（关键：这里就是每个像素一个 a、一个 b）
        a = 1.0 + self.a_scale * torch.tanh(a_raw)
        b = self.b_scale * torch.tanh(b_raw)
        return a, b, gate

class CascadeRefine3Stage(nn.Module):
    def __init__(self, base_channels=32, a_scale=0.2, b_scale=0.2, use_delta_as_input=True, channels=1):
        super().__init__()
        self.stage1 = Stage1DeltaNet(base_channels=base_channels, channels=channels)
        self.stage2 = ABAdjustNet(base_channels=base_channels, a_scale=a_scale, b_scale=b_scale,
                                 use_delta_as_input=use_delta_as_input, channels=channels)
        self.stage3 = ABAdjustNet(base_channels=base_channels, a_scale=a_scale, b_scale=b_scale,
                                 use_delta_as_input=use_delta_as_input, channels=channels)

    def forward(self, x_raw, x_coarse0):
        # ---- Stage 1 ----
        delta1, gate1 = self.stage1(x_raw, x_coarse0)     # gated refine delta
        x_coarse1 = x_coarse0 + delta1                    # 更新 coarse（相加）
        # 这里 x_coarse1 就是你说的 “refine output + coarse input 的和”

        # ---- Stage 2 ----
        a2, b2, gate2 = self.stage2(x_raw, x_coarse1, delta1)  # per-pixel a,b,gate
        proposed_delta2 = a2 * delta1 + b2
        delta2 = delta1 + gate2 * (proposed_delta2 - delta1)
        x_coarse2 = x_coarse0 + delta2                    # 更新 coarse（相加得到下一轮 coarse）

        # ---- Stage 3 ----
        a3, b3, gate3 = self.stage3(x_raw, x_coarse2, delta2)
        proposed_delta3 = a3 * delta2 + b3
        delta3 = delta2 + gate3 * (proposed_delta3 - delta2)
        x_final = x_coarse0 + delta3                      # 最终输出图

        return {
            "delta1": delta1, "coarse1": x_coarse1,
            "delta2": delta2, "coarse2": x_coarse2,
            "delta3": delta3, "final":  x_final,
            "a2": a2, "b2": b2, "gate1": gate1, "gate2": gate2,
            "a3": a3, "b3": b3, "gate3": gate3,
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
    gx = F.conv2d(x, sobel_x.repeat(x.shape[1],1,1,1), padding=1, groups=x.shape[1])
    gy = F.conv2d(x, sobel_y.repeat(x.shape[1],1,1,1), padding=1, groups=x.shape[1])
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
    g, _, _ = compute_sobel_grad(target, eps=eps, normalize=normalize_grad)
    g = g.mean(dim=1, keepdim=True)  # (B,1,H,W)

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
    grad_diff = ((gpx - gtx).abs() + (gpy - gty).abs()).mean(dim=1, keepdim=True)
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



