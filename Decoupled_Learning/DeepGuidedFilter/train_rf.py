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

class EdgeAwareRefineNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        """
        in_channels = 3 (Raw + Coarse + EdgeMap)
        结构：4层卷积的小网络
        """
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

            nn.Conv2d(base_channels, 1, kernel_size=1, bias=False)
        )

    def forward(self, x_raw, x_coarse):
        """
        x_raw:    (B, 1, H, W) - 带有丰富细节但有噪声的原始图
        x_coarse: (B, 1, H, W) - 经过前置滤波的干净图（但边缘可能糊了）
        """
        g, _, _ = compute_sobel_grad(x_coarse, normalize=True)
        inp = torch.cat([x_raw, x_coarse, g], dim=1) 
        delta = self.net(inp)
        out = x_coarse + delta      
        return out
    
# 计算Sobel梯度
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

def edge_guided_loss(pred, input, target,
                     thresh_mode='percentile',  # 'fixed' or 'percentile'
                     thresh=0.6,               # fixed 时用；或 percentile 的分位，如 0.9
                     dilate=1,                 # 边缘膨胀半径（像素），0 表示不膨胀
                     loss_type='l1',           # 'l1' | 'l2' | 'huber'
                     eps=1e-6,
                     normalize_grad=True):
    """
    pred, target: BxCxHxW，取值范围不限（建议都已归一到稳定范围）
    thresh_mode:
        - 'fixed'       : 使用固定阈值（若 normalize_grad=True，则阈值以[0,1]为尺度）
        - 'percentile'  : 每张图按分位数自适应阈值（推荐），thresh取0~1之间的分位比如0.9
    dilate: >0 会用 max-pool 膨胀边缘，扩大约束区域
    """
    assert pred.shape == target.shape, "pred/target 形状必须一致"
    # 1) 计算 input 的梯度（决定“哪里是边缘”）
    g,_,_ = compute_sobel_grad(target, eps=eps, normalize=normalize_grad)  # Bx1xHxW

    # 2) 生成边缘掩码
    if thresh_mode == 'fixed':
        T = torch.as_tensor(thresh, device=pred.device, dtype=pred.dtype)
        mask = (g > T).float()
    elif thresh_mode == 'percentile':
        # 每张图按分位数得到阈值（更鲁棒）
        q = torch.quantile(g.flatten(1), torch.as_tensor(thresh, device=g.device), dim=1)
        T = q.view(-1, 1, 1, 1)  # Bx1x1x1
        mask = (g > T).float()
    else:
        raise ValueError("thresh_mode must be 'fixed' or 'percentile'")

    # 3) 可选：扩边（膨胀），让边缘周围也参与约束
    if dilate and dilate > 0:
        k = dilate*2 + 1
        mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=dilate)

    # 4) 计算 masked loss（仅在边缘上）
    per_pix = per_channel_loss(pred, target, loss_type=loss_type)
    pix_loss = masked_reduce(per_pix, mask, eps=eps, reduction='mean')

    # --------  梯度域逐像素差异 --------
    gp, gpx, gpy = compute_sobel_grad(pred,   eps=eps, normalize=normalize_grad)  # (B,1,H,W)
    gt, gtx, gty = compute_sobel_grad(target, eps=eps, normalize=normalize_grad)  # (B,1,H,W)

    grad_loss = (gpx - gtx).abs() + (gpy - gty).abs()
    grad_loss = grad_loss * mask
    denom = mask.sum().clamp_min(eps)  # 防止除零
    grad_loss = grad_loss.sum() / denom

    # -------- 融合 --------
    eps = 1e-8  # 防止除零
    # 动态比例：让 grad_loss 和 pix_loss 尺度一致
    total = pix_loss \
        + grad_loss / ((grad_loss / (pix_loss + eps)).detach())

    return total

###### 训练输入 #############
origin_input_train = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/traininput'
input_folder_train = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/train_input_refine_circle'
input_folder_train_label = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/trainlabel'

###### 测试输入 ##############
origin_input_test = '/mnt/d/nuoqi/S2R-HDR/data/Single-Image-HDR/testinput'
input_folder_test = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/result/rf_output_singlehdr'
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

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0,
                          collate_fn=var_custom_collate, pin_memory=True, drop_last=True)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                          collate_fn=var_custom_collate, pin_memory=True, drop_last=False)

# --- 在每个 epoch 结束后保存 ---
def save_checkpoint(epoch, best_loss, is_best=False):
    # 保存完整状态，方便续训
    full_path = os.path.join(save_dir, "refine_filter_full_mix_circle.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss
    }, full_path)
    print(f"Checkpoint saved: {full_path}")

    # 保存纯模型 (方便推理或部署)
    last_model_path = os.path.join(save_dir, "refine_filter_last_mix_circle.pth")
    torch.save(model.state_dict(), last_model_path)
    print(f"Last model weights saved: {last_model_path}")

    # 如果是最佳模型，另存一份
    if is_best:
        best_model_path = os.path.join(save_dir, f"refine_filter_best_mix_circle.pth")
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
task_name = "refine_filter_mix_circle"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/log_dir", f"{task_name}_{timestamp}")
writer = SummaryWriter(log_dir=log_dir)

######### psnr 计算 ###########
def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1 / mse)
    return psnr, mse

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
device = torch.device('cuda:0')
model = EdgeAwareRefineNet(in_channels=3)
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

            batch_input_origin = crop_to_even(batch_input_origin)
            batch_input        = crop_to_even(batch_input)
            batch_label        = crop_to_even(batch_label)
            torch.cuda.synchronize(); t1 = time.time()

            batch_input_origin, batch_input, batch_label = batch_input_origin.to(device), batch_input.to(device), batch_label.to(device)

            # ----- 输入网络进行处理 ------ #
            #batch_input_masked = batch_input_masked - 0.5
            batch_input_origin = batch_input_origin - 0.5  
            batch_input        = batch_input - 0.5       # coarse filtered img   
            batch_label        = batch_label - 0.5        
                        
            #t1 = time.time()
            # 前向
            output = model(batch_input_origin, batch_input)
            torch.cuda.synchronize(); t3 = time.time()

            loss = edge_guided_loss(output, batch_input,  batch_label, 
                               thresh_mode='percentile',  # 推荐
                               thresh=0.6,                # 
                               dilate=0,                  # 让边缘区域更宽一点
                               loss_type='l1',
                               normalize_grad=True)
            # ----- 保存mask结果看看是否符合预期 ------ #
            # save_output_dir = 'D:/nuoqi/halo-free/Decoupled_Learning/HDRv1/train_vis/refinev6'
            # if batch_idx == len(train_loader) // 2:
            #     save_batch_images(output, save_output_dir, prefix="pred", epoch=epoch, batch_idx=batch_idx)

            #print(stats)

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(); t4 = time.time()

            output      = output + 0.5
            output      = torch.clamp(output, 0, 1)
            batch_label = batch_label + 0.5
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

                # ----- 对输入网络的数据进行处理 ----- #
                test_input_origin = test_input_origin - 0.5
                test_input = test_input - 0.5
                test_label = test_label - 0.5
                test_output = model(test_input_origin, test_input)

                loss = edge_guided_loss(test_output, test_input, test_label,
                               thresh_mode='percentile',  # 推荐
                               thresh=0.05,                # 用 top 10% 强边缘
                               dilate=1,                  # 让边缘区域更宽一点
                               loss_type='l1',
                               normalize_grad=True)


                test_output = test_output + 0.5
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











