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

class GuidedRefineBlock(nn.Module):
    def __init__(self, base_channels=32, feat_channels=32, a_scale=0.2, b_scale=0.2, use_feat=True):
        super().__init__()
        self.a_scale = a_scale
        self.b_scale = b_scale
        self.feat_channels = feat_channels
        self.use_feat = use_feat

        in_ch = 3 + (feat_channels if use_feat else 0)  # raw + coarse + grad(raw) + feat_prev(optional)

        self.feat_net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

        self.ab_head = nn.Conv2d(feat_channels, 2, 1, bias=True)  # (a_raw, b_raw)

    def forward(self, x_raw, x_coarse_k, feat_prev=None):
        g, _, _ = compute_sobel_grad(x_raw, normalize=True)

        if self.use_feat:
            if feat_prev is None:
                feat_prev = torch.zeros(
                    x_raw.size(0), self.feat_channels, x_raw.size(2), x_raw.size(3),
                    device=x_raw.device, dtype=x_raw.dtype
                )
            inp = torch.cat([x_raw, x_coarse_k, g, feat_prev], dim=1)
        else:
            inp = torch.cat([x_raw, x_coarse_k, g], dim=1)

        feat = self.feat_net(inp)
        ab = self.ab_head(feat)
        a_raw = ab[:, 0:1]
        b_raw = ab[:, 1:2]

        a = self.a_scale * torch.tanh(a_raw)
        b = self.b_scale * torch.tanh(b_raw)
        return a, b, feat

class CascadeRefine3Stage(nn.Module):
    def __init__(self, base_channels=32, feat_channels=32, a_scale=0.2, b_scale=0.2, use_feat=True):
        super().__init__()
        self.stage1 = GuidedRefineBlock(
            base_channels=base_channels,
            feat_channels=feat_channels,
            a_scale=a_scale,
            b_scale=b_scale,
            use_feat=use_feat,
        )
        self.stage2 = GuidedRefineBlock(
            base_channels=base_channels,
            feat_channels=feat_channels,
            a_scale=a_scale,
            b_scale=b_scale,
            use_feat=use_feat,
        )
        self.stage3 = GuidedRefineBlock(
            base_channels=base_channels,
            feat_channels=feat_channels,
            a_scale=a_scale,
            b_scale=b_scale,
            use_feat=use_feat,
        )

    def forward(self, x_raw, x_coarse0):
        # ---- Stage 1 ----
        a1, b1, feat1 = self.stage1(x_raw, x_coarse0, None)
        delta1 = a1 * x_raw + b1
        x_coarse1 = x_coarse0 + delta1

        # ---- Stage 2 ----
        a2, b2, feat2 = self.stage2(x_raw, x_coarse1, feat1)
        delta2 = a2 * x_raw + b2
        x_coarse2 = x_coarse1 + delta2

        # ---- Stage 3 ----
        a3, b3, feat3 = self.stage3(x_raw, x_coarse2, feat2)
        delta3 = a3 * x_raw + b3
        x_final = x_coarse2 + delta3

        return {
            "delta1": delta1, "coarse1": x_coarse1,
            "delta2": delta2, "coarse2": x_coarse2,
            "delta3": delta3, "final":  x_final,
            "a1": a1, "b1": b1,
            "a2": a2, "b2": b2,
            "a3": a3, "b3": b3,
            "feat1": feat1, "feat2": feat2, "feat3": feat3,
        }

def cascade_loss(outputs, x_coarse0, gt,
                 thresh_mode='percentile', thresh=0.6, dilate=0,
                 loss_type='l1', normalize_grad=True):
    delta_gt1 = gt - x_coarse0
    delta_gt2 = gt - outputs["coarse1"]
    delta_gt3 = gt - outputs["coarse2"]

    l1 = edge_guided_loss(outputs["delta1"],  delta_gt1,
                          thresh_mode=thresh_mode, thresh=thresh, dilate=dilate,
                          loss_type=loss_type, normalize_grad=normalize_grad)
    l2 = edge_guided_loss(outputs["delta2"], delta_gt2,
                          thresh_mode=thresh_mode, thresh=thresh, dilate=dilate,
                          loss_type=loss_type, normalize_grad=normalize_grad)
    l3 = edge_guided_loss(outputs["delta3"], delta_gt3,
                          thresh_mode=thresh_mode, thresh=thresh, dilate=dilate,
                          loss_type=loss_type, normalize_grad=normalize_grad)

    return 0.33*l1 + 0.33*l2 + 0.33*l3, (l1, l2, l3)

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

def edge_guided_loss(
    pred, target,
    thresh_mode='percentile',
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
    else:
        raise ValueError("thresh_mode must be 'fixed' or 'percentile'")

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

    # 4) dynamic align (your style): align each aux to edge_pix
    #    （等价于给 edge_grad 和 smooth_pix 一个“自动权重”，让它们量级≈edge_pix）
    total = edge_pix \
        + edge_grad  / ((edge_grad  / (edge_pix + eps)).detach()) \
        + smooth_pix / ((smooth_pix / (edge_pix + eps)).detach())

    return total

###### 训练输入 #############
origin_input_train = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/traininput'
input_folder_train = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/train_rf_3stage_input'
input_folder_train_label = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/trainlabel'

###### 测试输入 ##############
origin_input_test = '/mnt/d/nuoqi/S2R-HDR/data/Single-Image-HDR/testinput'
input_folder_test = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/test_rf_3stage_input'
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
    # # 保存完整状态，方便续训
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
        best_model_path = os.path.join(save_dir, f"rf_3stage_best_t0.8_rf_v2.pth")
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
task_name = "rf_3stage_thresh0.8_rf_v2"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("/mnt/d/nuoqi/halo-free/Decoupled_Learning/DeepGuidedFilter/log_dir", f"{task_name}_{timestamp}")
writer = SummaryWriter(log_dir=log_dir)

######### psnr 计算 ###########
def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1 / mse)
    return psnr, mse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda:0')
model = CascadeRefine3Stage(base_channels=32, feat_channels=32, a_scale=0.2, b_scale=0.2, use_feat=True)
model = torch.nn.DataParallel(model).to(device)
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
            torch.cuda.synchronize(); t0 = time.time()

            batch_input_origin = crop_to_even(batch_input_origin)
            batch_input        = crop_to_even(batch_input)
            batch_label        = crop_to_even(batch_label)
            torch.cuda.synchronize(); t1 = time.time()

            batch_input_origin, batch_input, batch_label = batch_input_origin.to(device), batch_input.to(device), batch_label.to(device)

            # ----- 输入网络进行处理 ------ #
            batch_input_origin = batch_input_origin - 0.5
            batch_input        = batch_input - 0.5       # coarse filtered img
            batch_label        = batch_label - 0.5

            # 前向
            outputs = model(batch_input_origin, batch_input)
            torch.cuda.synchronize(); t3 = time.time()

            loss, (l1, l2, l3) = cascade_loss(
                outputs, batch_input, batch_label,
                thresh_mode='percentile',
                thresh=0.8,
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
            for test_input_origin, test_input, test_label in test_loader:
                test_input_origin = crop_to_even(test_input_origin)
                test_input = crop_to_even(test_input)
                test_label = crop_to_even(test_label)

                test_input_origin, test_input, test_label = test_input_origin.to(device), test_input.to(device), test_label.to(device)

                # ----- 对输入网络的数据进行处理 ----- #
                test_input_origin = test_input_origin - 0.5
                test_input = test_input - 0.5
                test_label = test_label - 0.5
                test_outputs = model(test_input_origin, test_input)

                loss, _ = cascade_loss(
                    test_outputs, test_input, test_label,
                    thresh_mode='percentile',
                    thresh=0.8,
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
