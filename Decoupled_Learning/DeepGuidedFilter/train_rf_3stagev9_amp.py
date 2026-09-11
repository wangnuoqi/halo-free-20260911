import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast  # 引入混合精度

import os, datetime
from tqdm import tqdm
import cv2
import numpy as np
import time
import imageio

# ==========================================
# 核心：解耦 QKV 的方向性 Transformer (Local Window + 距离阻断版)
# ==========================================
class DirectionalTransformer(nn.Module):
    def __init__(self, in_channels=1, dim=16, direction='row', window_size=15):
        super().__init__()
        self.direction = direction
        self.dim = dim
        self.window_size = window_size
        self.pad = window_size // 2
        
        # 1. 结构化特征提取：用 1D 卷积取代 Linear，解决 Rank-1 崩塌
        # 行方向看水平邻域，列方向看垂直邻域，提取梯度的“局部结构特征”
        if direction == 'row':
            self.g_conv = nn.Conv2d(1, dim * 2, kernel_size=(1, 5), padding=(0, 2))
        else:
            self.g_conv = nn.Conv2d(1, dim * 2, kernel_size=(5, 1), padding=(2, 0))
            
        # V 的映射
        self.wv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 2. 距离惩罚项 (Relative Positional Penalty)
        # 初始化为一个随距离线性衰减的负值 (如: [-3.5, -3.0, ... 0 ... -3.0, -3.5])
        # 距离越远，初始 attention score 被扣分越多，阻断越强！
        init_bias = -torch.abs(torch.arange(-self.pad, self.pad + 1, dtype=torch.float32)) * 0.5
        self.rel_pos_bias = nn.Parameter(init_bias.view(1, 1, 1, 1, self.window_size))

    def forward(self, v_in, g_in):
        B, C, H, W = v_in.shape
        
        # 提取结构化梯度特征 -> Q, K
        g_feat = self.g_conv(g_in) # (B, 2*dim, H, W)
        q, k = torch.split(g_feat, self.dim, dim=1) # each (B, dim, H, W)
        v = self.wv(v_in) # (B, C, H, W)
        
        # 使用 Unfold 提取局部滑动窗口
        if self.direction == 'row':
            # 在 W 维度 padding 并展开
            k_pad = F.pad(k, (self.pad, self.pad, 0, 0), mode='replicate')
            v_pad = F.pad(v, (self.pad, self.pad, 0, 0), mode='replicate')
            
            k_unfold = k_pad.unfold(3, self.window_size, 1) # (B, dim, H, W, K)
            v_unfold = v_pad.unfold(3, self.window_size, 1) # (B, C, H, W, K)
            
            q_r = q.permute(0, 2, 3, 1).unsqueeze(3)        # (B, H, W, 1, dim)
            k_r = k_unfold.permute(0, 2, 3, 1, 4)           # (B, H, W, dim, K)
            
        else:
            # 在 H 维度 padding 并展开
            k_pad = F.pad(k, (0, 0, self.pad, self.pad), mode='replicate')
            v_pad = F.pad(v, (0, 0, self.pad, self.pad), mode='replicate')
            
            k_unfold = k_pad.unfold(2, self.window_size, 1) # (B, dim, H, W, K)
            v_unfold = v_pad.unfold(2, self.window_size, 1) # (B, C, H, W, K)
            
            q_r = q.permute(0, 2, 3, 1).unsqueeze(3)        # (B, H, W, 1, dim)
            k_r = k_unfold.permute(0, 2, 3, 1, 4)           # (B, H, W, dim, K)

        # 点积计算 Attention Score (此时网络只会对比局部 K 个像素)
        attn = torch.matmul(q_r, k_r) * (self.dim ** -0.5)  # (B, H, W, 1, K)
        
        # 【核心阻断】：加上随距离衰减的惩罚项
        attn = attn + self.rel_pos_bias
        
        # 恢复 Hard Softmax，实现跨边缘的断崖式权重清零
        attn = F.softmax(attn, dim=-1) # (B, H, W, 1, K)
        
        # 与 V 加权求和
        if self.direction == 'row':
            v_r = v_unfold.permute(0, 2, 3, 4, 1) # (B, H, W, K, C)
        else:
            v_r = v_unfold.permute(0, 2, 3, 4, 1) # (B, H, W, K, C)
            
        out = torch.matmul(attn, v_r) # (B, H, W, 1, C)
        out = out.squeeze(3).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        return out

class QWLS_Block(nn.Module):
    def __init__(self, in_channels=1, dim=16, window_size=15):
        super().__init__()
        self.row_trans = DirectionalTransformer(in_channels, dim, direction='row', window_size=window_size)
        self.col_trans = DirectionalTransformer(in_channels, dim, direction='col', window_size=window_size)
        
        # 聚合层：利用 1x1 卷积动态分配权重，再用 3x3 卷积让残差图更平滑
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, p_in, gx, gy):
        p_row = self.row_trans(p_in, gx)
        p_col = self.col_trans(p_in, gy)
        
        concat_features = torch.cat([p_in, p_row, p_col], dim=1)
        residual = self.fusion(concat_features)
        
        p_out = p_in + residual
        return p_out

class DecoupledQWLSNet(nn.Module):
    def __init__(self, in_channels=1, dim=16, stages=3):
        super().__init__()
        # 设置局域窗口大小为 15 (左右各看 7 个像素)
        self.stages = nn.ModuleList([QWLS_Block(in_channels, dim, window_size=15) for _ in range(stages)])

    def forward(self, x_raw, x_coarse0):
        # 提取全局高频向导
        _, gx, gy = compute_sobel_grad(x_raw, normalize=True)

        p = x_coarse0
        outputs = {}
        
        for i, stage in enumerate(self.stages):
            p = stage(p, gx, gy)
            outputs[f"coarse{i+1}"] = p
            
        outputs["final"] = p
        return outputs


# ==========================================
# 损失函数 & 工具集：Label GT + Otsu 阈值 (带安全阈值修改)
# ==========================================
def compute_sobel_grad(x, eps=1e-4, normalize=False):
    sobel_x = torch.tensor([[[[-1., 0., 1.],
                              [-2., 0., 2.],
                              [-1., 0., 1.]]]], device=x.device, dtype=x.dtype)
    sobel_y = torch.tensor([[[[-1., -2., -1.],
                              [ 0.,  0.,  0.],
                              [ 1.,  2.,  1.]]]], device=x.device, dtype=x.dtype)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    g = torch.sqrt(gx*gx + gy*gy + eps)
    
    if normalize:
        g = g / (g.amax(dim=(2,3), keepdim=True) + eps)
        gx = gx.abs() / (gx.abs().amax(dim=(2,3), keepdim=True) + eps)
        gy = gy.abs() / (gy.abs().amax(dim=(2,3), keepdim=True) + eps)
    else:
        gx = gx.abs()
        gy = gy.abs()
        
    return g, gx, gy 

def compute_otsu_threshold_batch(tensor, bins=256):
    B = tensor.shape[0]
    x = tensor.view(B, -1)
    
    img_min = x.min(dim=1, keepdim=True)[0]
    img_max = x.max(dim=1, keepdim=True)[0]
    
    diff = img_max - img_min
    diff = torch.where(diff < 1e-4, torch.ones_like(diff), diff) 
    
    x_norm = ((x - img_min) / diff * (bins - 1)).round().long()
    x_norm = x_norm.clamp(0, bins - 1)
    
    hist = F.one_hot(x_norm, num_classes=bins).sum(dim=1).float()
    
    weight_back = torch.cumsum(hist, dim=1) 
    total_pixels = x.shape[1]
    weight_fore = total_pixels - weight_back 
    
    bins_arange = torch.arange(bins, device=tensor.device, dtype=torch.float32).view(1, -1)
    sum_back = torch.cumsum(hist * bins_arange, dim=1) 
    sum_total = sum_back[:, -1:] 
    
    weight_back_safe = weight_back.clamp(min=1e-4)
    weight_fore_safe = weight_fore.clamp(min=1e-4)
    
    mean_back = sum_back / weight_back_safe
    mean_fore = (sum_total - sum_back) / weight_fore_safe
    
    var_between = weight_back * weight_fore * (mean_back - mean_fore) ** 2
    
    invalid_mask = (weight_back == 0) | (weight_fore == 0)
    var_between.masked_fill_(invalid_mask, 0.0)
    
    threshold_idx = var_between.argmax(dim=1).float() 
    thresholds = (threshold_idx.view(B, 1) / (bins - 1)) * (img_max - img_min) + img_min
    
    return thresholds.view(B, 1, 1, 1)

def per_channel_loss(pred, target, loss_type='l1'):
    if loss_type == 'l1': return (pred - target).abs()
    elif loss_type == 'l2': return (pred - target) ** 2
    elif loss_type == 'huber': return F.smooth_l1_loss(pred, target, reduction='none')
    else: raise ValueError("loss_type must be 'l1' | 'l2' | 'huber'")

def masked_reduce(per, mask=None, eps=1e-4, reduction='mean'):
    if mask is not None:
        per = per * mask
        if reduction == 'mean':
            denom = mask.sum().clamp_min(eps)
            return per.sum() / denom
        elif reduction == 'sum': return per.sum()
        else: return per
    else:
        if reduction == 'mean': return per.mean()
        elif reduction == 'sum': return per.sum()
        else: return per

def edge_guided_loss(pred, target, thresh_mode='otsu', thresh=0.8, dilate=0, loss_type='l1', eps=1e-4, normalize_grad=True, detach_mask=True):
    assert pred.shape == target.shape
    g, _, _ = compute_sobel_grad(target, eps=eps, normalize=normalize_grad) 

    if thresh_mode == 'fixed':
        T = torch.as_tensor(thresh, device=pred.device, dtype=pred.dtype)
        edge_mask = (g > T).float()
    elif thresh_mode == 'percentile':
        q = torch.quantile(g.flatten(1), torch.as_tensor(thresh, device=g.device), dim=1)
        edge_mask = (g > q.view(-1,1,1,1)).float()
    elif thresh_mode == 'otsu':
        T = compute_otsu_threshold_batch(g)
        edge_mask = (g > T).float()
    else:
        raise ValueError("thresh_mode 错误")

    if dilate and dilate > 0:
        k = dilate * 2 + 1
        edge_mask = F.max_pool2d(edge_mask, kernel_size=k, stride=1, padding=dilate)

    if detach_mask:
        edge_mask = edge_mask.detach()

    smooth_mask = (1.0 - edge_mask).clamp(0.0, 1.0)
    per_pix = per_channel_loss(pred, target, loss_type=loss_type) 
    if per_pix.dim() == 4 and per_pix.shape[1] > 1:
        per_pix = per_pix.mean(dim=1, keepdim=True) 

    edge_pix = masked_reduce(per_pix, edge_mask, eps=eps, reduction='mean')
    smooth_pix = masked_reduce(per_pix, smooth_mask, eps=eps, reduction='mean')

    _, gpx, gpy = compute_sobel_grad(pred, eps=eps, normalize=normalize_grad)
    _, gtx, gty = compute_sobel_grad(target, eps=eps, normalize=normalize_grad)
    grad_diff = (gpx - gtx).abs() + (gpy - gty).abs()
    edge_grad = masked_reduce(grad_diff, edge_mask, eps=eps, reduction='mean')

    total = edge_pix \
          + edge_grad  / ((edge_grad  / (edge_pix + eps)).detach()) \
          + smooth_pix / ((smooth_pix / (edge_pix + eps)).detach())
    return total

def cascade_loss(outputs, gt, thresh_mode='otsu', thresh=0.8, dilate=0, loss_type='l1', normalize_grad=True):
    l1 = edge_guided_loss(outputs["coarse1"], gt, thresh_mode=thresh_mode, thresh=thresh, dilate=dilate, loss_type=loss_type, normalize_grad=normalize_grad)
    l2 = edge_guided_loss(outputs["coarse2"], gt, thresh_mode=thresh_mode, thresh=thresh, dilate=dilate, loss_type=loss_type, normalize_grad=normalize_grad)
    l3 = edge_guided_loss(outputs["final"],   gt, thresh_mode=thresh_mode, thresh=thresh, dilate=dilate, loss_type=loss_type, normalize_grad=normalize_grad)
    return 0.33*l1 + 0.33*l2 + 0.33*l3, (l1, l2, l3)


# ==========================================
# 数据加载 & 工具
# ==========================================
###### 训练/测试输入路径 #############
origin_input_train = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/traininput'
input_folder_train = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/train_refine_input'
input_folder_train_label = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/trainlabel'

origin_input_test = '/mnt/d/nuoqi/S2R-HDR/data/Single-Image-HDR/testinput'
input_folder_test = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/test_refine_input'
input_folder_test_label = '/mnt/d/nuoqi/S2R-HDR/data/Single-Image-HDR/testlabel'

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
        input_img      = cv2.imread(self.input_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        label          = cv2.imread(self.label_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        input_hr_h, input_hr_w = input_img.shape

        return {
            'origin_input_hr': torch.from_numpy(origin_input).unsqueeze(0).unsqueeze(0),
            'input_hr': torch.from_numpy(input_img).unsqueeze(0).unsqueeze(0),
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
        batch_input_origin[idx] = item['origin_input_hr'][:, :, :min_h_hr, :min_w_hr]
        batch_input[idx] = item['input_hr'][:, :, :min_h_hr, :min_w_hr]
        batch_target_images[idx] = item['target_img'][:, :, :min_h_hr, :min_w_hr]

    return batch_input_origin, batch_input, batch_target_images

train_dataset = GuidedFilterDataset(origin_input_train, input_folder_train, input_folder_train_label)
test_dataset = GuidedFilterDataset(origin_input_test, input_folder_test, input_folder_test_label)

def crop_to_even(tensor):
    _, _, H, W = tensor.shape
    new_H = H if H % 2 == 0 else H - 1
    new_W = W if W % 2 == 0 else W - 1
    return tensor[:, :, :new_H, :new_W]

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=24, collate_fn=var_custom_collate, pin_memory=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=var_custom_collate, pin_memory=False, drop_last=False)

def save_checkpoint(epoch, best_loss, is_best=False):
    if is_best:
        best_model_path = os.path.join(save_dir, f"rf_3stage_qwls_local_window_v9.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"*** Best model updated: {best_model_path} (Test MSE: {best_loss:.6f}) ***")

def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1 / mse)
    return psnr, mse

# ==========================================
# 训练主循环
# ==========================================
task_name = "rf_3stage_qwls_local_window_v9"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("/mnt/d/nuoqi/halo-free/Decoupled_Learning/DeepGuidedFilter/log_dir", f"{task_name}_{timestamp}")
writer = SummaryWriter(log_dir=log_dir)

device_ids = [0, 1]
device = torch.device(f'cuda:{device_ids[0]}')

# 使用新架构：in_channels=1, hidden_dim=16, 3 层级联
model = DecoupledQWLSNet(in_channels=1, dim=16, stages=3)
visible_gpus = torch.cuda.device_count()
if visible_gpus <= max(device_ids):
    raise RuntimeError(
        f"Requested GPUs {device_ids}, but only {visible_gpus} CUDA device(s) are visible."
    )
model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0]).to(device)
print(f"[Info] DataParallel device_ids={device_ids}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

scaler = GradScaler()
best_loss = float("inf")
save_dir = "/mnt/d/nuoqi/halo-free/Decoupled_Learning/DeepGuidedFilter/models_retrain"
os.makedirs(save_dir, exist_ok=True)
epochs = 80

if __name__ == "__main__":
    for epoch in tqdm(range(epochs), desc="Training Progress", dynamic_ncols=True):
        model.train()
        epoch_mse = 0
        epoch_psnr = 0
        for batch_idx, (batch_input_origin, batch_input, batch_label) in enumerate(train_loader):
            batch_input_origin = crop_to_even(batch_input_origin).to(device)
            batch_input        = crop_to_even(batch_input).to(device)
            batch_label        = crop_to_even(batch_label).to(device)

            batch_input_origin = batch_input_origin - 0.5
            batch_input        = batch_input - 0.5
            batch_label        = batch_label - 0.5

            optimizer.zero_grad()

            with autocast():
                outputs = model(batch_input_origin, batch_input)
                
            # 转为 FP32 计算 Loss，防止 NaN
            outputs_fp32 = {k: v.float() for k, v in outputs.items()}
            batch_label_fp32 = batch_label.float()

            loss, (l1, l2, l3) = cascade_loss(
                outputs_fp32, batch_label_fp32,
                thresh_mode='otsu',
                dilate=0,
                loss_type='l1',
                normalize_grad=True
            )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            output_final = outputs["final"].detach().float() + 0.5
            output_final = torch.clamp(output_final, 0, 1)
            b_label = batch_label.float() + 0.5
            
            psnr, mse = calc_psnr(output_final, b_label)
            if batch_idx % 200 == 0:
                print(f"iter MSE loss: {mse:.6f}")
                
            epoch_mse += mse
            epoch_psnr += psnr
            
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 50 == 0:
                writer.add_scalar('train/mse', mse, global_step)
                writer.add_scalar('train/psnr', psnr, global_step)
                writer.add_scalar('train/l1', l1.item(), global_step)

        avg_epoch_mse = epoch_mse / len(train_loader)
        avg_epoch_psnr = epoch_psnr / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - MSE: {avg_epoch_mse:.6f} - PSNR: {avg_epoch_psnr:.2f} dB ")
        writer.add_scalar('train/mse_epoch', avg_epoch_mse, epoch)
        writer.add_scalar('train/psnr_epoch', avg_epoch_psnr, epoch)
        scheduler.step()

        # 测试验证阶段
        model.eval()
        test_mse = 0
        test_psnr = 0
        with torch.no_grad():
            for test_input_origin, test_input, test_label in test_loader:
                test_input_origin = crop_to_even(test_input_origin).to(device)
                test_input = crop_to_even(test_input).to(device)
                test_label = crop_to_even(test_label).to(device)

                test_input_origin = test_input_origin - 0.5
                test_input = test_input - 0.5
                test_label = test_label - 0.5
                
                with autocast():
                    test_outputs = model(test_input_origin, test_input)
                    
                test_outputs_fp32 = {k: v.float() for k, v in test_outputs.items()}
                test_label_fp32 = test_label.float()

                loss, _ = cascade_loss(
                    test_outputs_fp32, test_label_fp32,
                    thresh_mode='otsu',
                    dilate=0,
                    loss_type='l1',
                    normalize_grad=True
                )

                test_output = test_outputs["final"].float() + 0.5
                test_output = torch.clamp(test_output, 0, 1)
                t_label = test_label.float() + 0.5
                
                psnr, mse = calc_psnr(test_output, t_label)
                test_mse += mse
                test_psnr += psnr

        avg_test_mse = test_mse / len(test_loader)
        avg_test_psnr = test_psnr / len(test_loader)

        print(f"Epoch {epoch+1}/{epochs} - Test MSE: {avg_test_mse:.6f} - Test PSNR: {avg_test_psnr:.2f} dB")
        writer.add_scalar('test/mse', avg_test_mse, epoch)
        writer.add_scalar('test/psnr', avg_test_psnr, epoch)

        is_best = avg_test_mse < best_loss
        if is_best:
            best_loss = avg_test_mse

        save_checkpoint(epoch, best_loss, is_best=is_best)

    writer.close()