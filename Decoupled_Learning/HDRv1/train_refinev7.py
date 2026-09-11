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
from guided_filter_pytorch.guided_filter import FastGuidedFilter
from guided_filter_pytorch.guided_filter import GuidedFilter
from guided_filter_pytorch.guided_filter import ConvGuidedFilter
import cv2
import numpy as np
import time
import math
import imageio
import cProfile
from torch.cuda.amp import autocast, GradScaler

class EdgeAwareRefineNet(nn.Module):
    """
    新流程：
      masked_x --(masked_encoder)--> feat --(1x1 conv)--> delta
      out = x + delta
    """
    def __init__(self,
                 in_channels=1,        # x 的通道数（通常 1）
                 base_channels=32):    # masked 分支特征宽度
        super().__init__()
        self.in_channels = in_channels

        # 分支A：仅对 masked_x 提特征
        self.masked_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # 将边缘特征映射回与 x 相同的通道数，便于做残差相加
        self.edge_proj = nn.Conv2d(base_channels, in_channels, kernel_size=1, bias=False)
        # 一般这里不再接 BN/激活，直接作为残差

    def forward(self, masked_x, x):
        """
        masked_x: (B, in_channels, H, W)
        x       : (B, in_channels, H, W)
        """
        feat  = self.masked_encoder(masked_x)     # (B, base_channels, H, W)
        delta = self.edge_proj(feat)              # (B, in_channels, H, W)
        out   = x + delta                         # 残差加和作为最终输出
        return out

# 计算Sobel梯度
# --------- 工具：Sobel 梯度幅值（对灰度图）用于生成mask---------
class EdgeComputation(nn.Module):
    def __init__(self):
        super(EdgeComputation, self).__init__()

    def forward(self, x):
        x_diffx = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])
        x_diffy = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])

        y = torch.zeros_like(x)
        y[:,:,:,1:] += x_diffx
        y[:,:,:,:-1] += x_diffx
        y[:,:,1:,:] += x_diffy
        y[:,:,:-1,:] += x_diffy
        y = torch.sum(y,1)
        y /= 4
        return y
edgeCompute = EdgeComputation()

@torch.no_grad()
def percentile_threshold(tensor, q=0.1):
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)  # 加一个 batch 维度变成 [1, C, H, W]
    B, C, H, W = tensor.shape
    flat = tensor.view(B, -1)
    thresh = torch.quantile(flat, q, dim=1, keepdim=True)  # shape: (B,1)
    return thresh.view(B, 1, 1, 1)  # 方便广播

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

def sobel_gradient_loss(pred: torch.Tensor,
                        target: torch.Tensor,
                        mode: str = 'vector',   # 'vector' 或 'magnitude'
                        reduction: str = 'mean',
                        eps: float = 1e-6) -> torch.Tensor:
    """
    pred, target: (B,C,H,W) 或 (B,1,H,W)
    mode='vector'  : L1(|∂x| + |∂y|)（推荐，方向+幅值都对齐）
    mode='magnitude': L1(|∇|)        （仅幅值）
    """
    pgx, pgy, pm = compute_sobel_grad(pred, eps)
    tgx, tgy, tm = compute_sobel_grad(target, eps)

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

def save_input_target(input_tensor, target_tensor,
                      save_dir="/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRV1/train_vis/refine_v7",
                      tag="",
                      makedirs=True):
    """
    input_tensor, target_tensor: (B, C, H, W) torch.Tensor
    save_dir: 保存目录
    tag: 文件名中附加标记，如 f"ep{epoch}_it{batch_idx}"
    """
    if makedirs:
        os.makedirs(save_dir, exist_ok=True)

    def to_01(t: torch.Tensor) -> torch.Tensor:
        t = t.detach().float().cpu()
        # 尝试自动归一到 [0,1]
        t_min, t_max = t.min().item(), t.max().item()
        if t_min < 0.0 and t_max <= 0.5 + 1e-6:      # 典型 [-0.5, 0.5]
            t = (t + 0.5).clamp(0, 1)
        elif t_max > 1.0 + 1e-6:                    # 可能是 0~255
            t = (t / 255.0).clamp(0, 1)
        else:
            t = t.clamp(0, 1)
        return t

    x = to_01(input_tensor)
    y = to_01(target_tensor)

    B = x.size(0)
    for i in range(B):
        stem = f"{tag}_s{i}" if tag else f"s{i}"
        save_image(x[i], os.path.join(save_dir, f"input_{stem}.png"))
        save_image(y[i], os.path.join(save_dir, f"target_{stem}.png"))

def compute_sobel_grad(x: torch.Tensor, eps: float = 1e-12, normalize: bool = True,
                       reduce: str = "l2"):
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    # Sobel 核（跟随输入 dtype/device）
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], device=device, dtype=dtype).view(1, 1, 3, 3)

    # 每个通道独立做梯度
    gx = F.conv2d(x, kx.repeat(C, 1, 1, 1), padding=1, groups=C)
    gy = F.conv2d(x, ky.repeat(C, 1, 1, 1), padding=1, groups=C)

    if reduce == "l1":
        mag_c = gx.abs() + gy.abs()            # (B,C,H,W)
    else:  # 'l2'
        mag_c = torch.sqrt(gx * gx + gy * gy + eps)

    # 按通道聚合到单通道幅值（也可以用 sum，这里用 mean 比较稳）
    mag = mag_c.mean(dim=1, keepdim=True)      # (B,1,H,W)

    if normalize:
        # 每张图内做 min-max 归一化到[0,1]，避免不同幅值量纲不一致
        m = mag.flatten(1)
        m_min = m.min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        m_max = m.max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        mag = (mag - m_min) / (m_max - m_min + eps)
        mag = mag.clamp(0, 1)

    return mag, gx, gy

def edge_guided_loss(pred, input, target,      
                     loss_type='l1',           # 'l1' | 'l2' | 'huber'
                     eps=1e-6):

    assert pred.shape == target.shape, "pred/target 形状必须一致"
    # 1) 计算 input 的梯度（决定“哪里是边缘”）----- 生成边缘掩码 #
    edge_input = edgeCompute(input)
    edge_target = edgeCompute(target)
    tau_input = percentile_threshold(edge_input, q=0.7)     # 去掉低 10%
    tau_target = percentile_threshold(edge_target, q=0.7)

    m_input  = (edge_input  > tau_input).float()
    m_target = (edge_target > tau_target).float()

    mask_on  = m_input * m_target             # 被 GT 支持的 input 边缘
    mask_off = m_input * (1.0 - m_target)      # 被 GT 否定的 input 边缘 → 假边
    
    # --------- 图片过程可视化代码 ----------- #
    from torchvision.utils import save_image
    save_dir = '/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/train_vis/refine'
    B = m_input.shape[0]
    K = min(1, B)
    save_tag=""
    mi = m_input.detach().cpu().clamp(0, 1)     # (B,1,H,W)
    mt = m_target.detach().cpu().clamp(0, 1)
    for i in range(K):

        stem = f"{save_tag}_s{i}" if save_tag else f"s{i}"
        save_image(mi[i], os.path.join(save_dir, f"m_on_{stem}.png"))
        save_image(mt[i], os.path.join(save_dir, f"m_off_{stem}.png"))
    

    # 2) 计算 masked loss（仅在边缘上）
    per_pix_on = per_channel_loss(pred, target, loss_type=loss_type)
    loss_on = masked_reduce(per_pix_on, mask_on, eps=eps, reduction='mean')

    per_pix_off = per_channel_loss(pred, input, loss_type=loss_type)
    loss_off = masked_reduce(per_pix_off, mask_off, eps=eps, reduction='mean')

    # 3) --------  梯度域逐像素差异 --------
    resid = pred - input
    g_resid, _, _ = compute_sobel_grad(resid, normalize=True)          # Bx1xHxW
    loss_off_grad = masked_reduce(g_resid, mask_off, eps=eps, reduction='mean')

    # —— 5) 可选：真边梯度对齐（形状更准） —— #
    grad_loss = sobel_gradient_loss(pred, target, mode='vector')       # Bx1xHxW 或标量
    grad_loss = masked_reduce(grad_loss, mask_on, eps=eps, reduction='mean')

    # -------- 融合 --------
    eps = 1e-8  # 防止除零
    loss = (
        loss_on
        + grad_loss    / ((grad_loss    / (loss_on + eps)).detach())
        + loss_off     / ((loss_off     / (loss_on + eps)).detach())
        + loss_off_grad/ ((loss_off_grad/ (loss_on + eps)).detach())
    )

    return loss, {
        'loss_on': loss_on.detach(),
        'loss_grad_on': grad_loss.detach(),
        'loss_off_pix': loss_off.detach(),
        'loss_off_grad': loss_off_grad.detach()
    }

###### 训练输入 #############
origin_input_train = '/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/input_train_log'
input_folder_train = '/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/refine_input_train'
input_folder_train_label = '/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/trainlabel_log'

###### 测试输入 ##############
origin_input_test = '/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/input_test_log'
input_folder_test = '/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/result/deepfilter_output'
input_folder_test_label = '/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/testlabel_log'

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

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0,
                          collate_fn=var_custom_collate, pin_memory=True, drop_last=True)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0,
                          collate_fn=var_custom_collate, pin_memory=True, drop_last=False)

# --- 在每个 epoch 结束后保存 ---
def save_checkpoint(epoch, best_loss, is_best=False):
    # 保存完整状态，方便续训
    full_path = os.path.join(save_dir, "refine_filter_full.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss
    }, full_path)
    print(f"Checkpoint saved: {full_path}")

    # 保存纯模型 (方便推理或部署)
    last_model_path = os.path.join(save_dir, "refine_filter_last.pth")
    torch.save(model.state_dict(), last_model_path)
    print(f"Last model weights saved: {last_model_path}")

    # 如果是最佳模型，另存一份
    if is_best:
        best_model_path = os.path.join(save_dir, f"refine_filter_best.pth")
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

        img_uint8 = (img_norm * 255).astype(np.uint8)

        filename = os.path.join(
            save_dir, f"{prefix}{epoch}_batch{batch_idx}_sample{i}.png"
        )
        imageio.imwrite(filename, img_uint8)
        print(f"[Saved] {filename}")

##### tensorboard 启动指令 #######
task_name = "refine_filter"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/log_dir", f"{task_name}_{timestamp}")
writer = SummaryWriter(log_dir=log_dir)

######### psnr 计算 ###########
def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1 / mse)
    return psnr, mse


device = torch.device('cuda:0')
model = EdgeAwareRefineNet(in_channels=1)
model = torch.nn.DataParallel(model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5) 
best_loss = float("inf")
save_dir = "/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/models_retrain"
epochs = 80

if __name__ == "__main__":
#def train_one_epoch():
######## -- 模型训练 -- #############
    for epoch in tqdm(range(epochs), desc="Training Progress", dynamic_ncols=True):
        model.train()
        epoch_mse = 0
        epoch_psnr = 0
        for batch_idx, (batch_input_origin, batch_input, batch_label) in enumerate(train_loader):
            torch.cuda.synchronize(); t0 = time.time()        
            #t0 = time.time()

            batch_input_origin = crop_to_even(batch_input_origin)
            batch_input = crop_to_even(batch_input)
            batch_label = crop_to_even(batch_label)
            torch.cuda.synchronize(); t1 = time.time()

            batch_input_origin, batch_input, batch_label = batch_input_origin.to(device), batch_input.to(device), batch_label.to(device)

            g = edgeCompute(batch_input_origin)

            thresh = 0.6
            q = torch.quantile(g.flatten(1), torch.as_tensor(thresh, device=g.device), dim=1)
            T = q.view(-1, 1, 1, 1)  # Bx1x1x1
            mask = (g > T).float()
            batch_input_masked = batch_input_origin * mask
            torch.cuda.synchronize(); t2 = time.time()
            # ----- 保存mask结果看看是否符合预期 ------ #
            save_mask_dir = '/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/train_vis/refine'
            if batch_idx == len(train_loader) // 2:
                save_batch_images(batch_input_masked, save_mask_dir, prefix="masked", epoch=epoch, batch_idx=batch_idx)

            # ----- 输入网络进行处理 ------ #
            batch_input_masked = batch_input_masked - 0.5
            batch_input = batch_input - 0.5
            batch_label = batch_label
            
            #t1 = time.time()
            # 前向
            output = model(batch_input_masked, batch_input)
            torch.cuda.synchronize(); t3 = time.time()

            loss, stats = edge_guided_loss(output, batch_input_origin,  batch_label, loss_type='l1', eps=1e-6)
            # ----- 保存mask结果看看是否符合预期 ------ #
            save_output_dir = '/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/train_vis/refine'
            if batch_idx == len(train_loader) // 2:
            #if batch_idx < 5:
                save_batch_images(output, save_output_dir, prefix="pred", epoch=epoch, batch_idx=batch_idx)

            #print(stats)
            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(); t4 = time.time()
            #print(f"[Batch {batch_idx}] crop: {(t1 - t0)*1000:.2f}ms | mask: {(t2 - t1)*1000:.2f}ms | forward: {(t3 - t2)*1000:.2f}ms | backward: {(t4 - t3)*1000:.2f}ms")

            # 计算 PSNR 和 MSE
            # if not torch.isfinite(output).all() or not torclh.isfinite(batch_label).all():
            #     print("Warning: NaN or Inf detected in output or labels!")
            #     print("Output stats:", output.min().item(), output.max().item(), output.mean().item())
            #     print("Label stats:", batch_label.min().item(), batch_label.max().item(), batch_label.mean().item())

            #     # 打印前几个值（方便观察是否过大）
            #     print("Sample output values:", output.view(-1)[:10])
            #     print("Sample label values:", batch_label.view(-1)[:10])

            psnr, mse = calc_psnr(output.detach(), batch_label)
            if batch_idx % 200 == 0:
                print(f"iter MSE loss: {mse:.6f}")
            epoch_mse += mse
            epoch_psnr += psnr
            # 每 50 step 记录到 TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 50 == 0:
                writer.add_scalar('train/mse', mse, global_step)
                writer.add_scalar('train/psnr', psnr, global_step)
            #t4 = time.time()
            #if batch_idx % 20 == 0:
            #    print(f"[Batch {batch_idx}] Data: {t1-t0:.3f}s | Forward: {t2-t1:.3f}s | Backward: {t3-t2:.3f}s | Log: {t4-t3:.3f}s", flush=True)
        #scheduler.step()
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

                test_input_origin, test_input, test_label = test_input_origin.to(device), test_input.to(device), test_label.to(device)

                g,_,_ = compute_sobel_grad(test_input_origin, eps=1e-6, normalize=True)
                thresh = 0.6
                q = torch.quantile(g.flatten(1), torch.as_tensor(thresh, device=g.device), dim=1)
                T = q.view(-1, 1, 1, 1)  # Bx1x1x1
                mask = (g > T).float()
                test_input_masked = test_input_origin * mask

                # ----- 对输入网络的数据进行处理 ----- #
                test_input_masked = test_input_masked - 0.5
                test_input = test_input - 0.5
                test_label = test_label
                test_output = model(test_input_masked, test_input)
                #test_output = model(test_input)

                #loss = edge_sensitive_loss(test_output, test_label, edge_weight=10.0)
                #loss = edge_position_loss(test_output, test_label, grad_thresh=0.1)
                loss, stats = edge_guided_loss(test_output, test_input_origin,   test_label, loss_type='l1', eps=1e-6)


                psnr, mse = calc_psnr(test_output, test_label)
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

# if __name__ == "__main__":
#     profiler = cProfile.Profile()
#     profiler.enable()

#     train_one_epoch()  # 或者你原来的训练 loop

#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats('cumulative')
#     stats.print_stats(20)  # 打印耗时前20的函数


  









