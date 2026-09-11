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

class EdgeAwareRefineNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super(EdgeAwareRefineNet, self).__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        residual = self.refine(x)
        refined_output = x + residual
        return refined_output

# 计算Sobel梯度
def compute_sobel_grad(img):
    sobel_x = torch.tensor([[[-1., 0., 1.], 
                                [-2., 0., 2.], 
                                [-1., 0., 1.]]], device=img.device).unsqueeze(0)
    sobel_y = torch.tensor([[[-1., -2., -1.], 
                                [0., 0., 0.], 
                                [1., 2., 1.]]], device=img.device).unsqueeze(0)

    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)
    grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    return grad

def edge_sensitive_loss(pred, target, edge_weight=10.0):
    # 梯度图作为权重
    target_grad = compute_sobel_grad(target)
    weight_map = 1 + edge_weight * target_grad  # 在边缘区域赋予更大的权重

    # 带权重的MSE损失
    loss = ((pred - target) ** 2) * weight_map
    return loss.mean()

# --- 自定义边缘强制拟合Loss函数 ---
def edge_position_loss(pred, target, grad_thresh=0.1):
    # 计算输入(pred)图像梯度
    input_grad = compute_sobel_grad(pred)

    # 获取梯度明显位置（二值mask），超过阈值为1，否则0
    edge_mask = (input_grad > grad_thresh).float()

    # 提取mask位置的pred和target像素值
    diff = (pred - target) * edge_mask

    # 计算仅在梯度显著位置的损失 (L1 loss 更稳定)
    loss = diff.abs().sum() / (edge_mask.sum() + 1e-6)

    return loss

###### 训练输入 #############
input_folder_train = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/refine_img_train'
input_folder_train_label = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRinput_label'

###### 测试输入 ##############
input_folder_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/refine_img_test'
input_folder_test_label = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/WLS-021544_V4'

##### - 数据加载 - #######
class GuidedFilterDataset(Dataset):
    def __init__(self, input_dir, label_dir):
        #self.x_lr_files = sorted([os.path.join(x_lr_dir, f) for f in os.listdir(x_lr_dir)])
        self.input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        print(f"INPUT: {len(self.input_files)}, Label: {len(self.label_files)}")

    def __len__(self):
        return min(len(self.input_files), len(self.label_files))

    def __getitem__(self, idx):
        #x_lr   = cv2.imread(self.x_lr_files[idx], cv2.IMREAD_GRAYSCALE).astype('float32')
        input   = cv2.imread(self.input_files[idx], cv2.IMREAD_GRAYSCALE).astype('float32')
        label  = cv2.imread(self.label_files[idx], cv2.IMREAD_GRAYSCALE).astype('float32')
        input_hr_h, input_hr_w = input.shape

        return {
            'input_hr': torch.from_numpy(input).unsqueeze(0).unsqueeze(0),
            'target_img': torch.from_numpy(label).unsqueeze(0).unsqueeze(0),
            'input_height_hr': input_hr_h,
            'input_width_hr': input_hr_w,
        }

def var_custom_collate(batch):
    min_h_hr = min([item['input_height_hr'] for item in batch])
    min_w_hr = min([item['input_width_hr'] for item in batch])

    batch_input = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    #batch_x_lr   = torch.zeros(len(batch), 1, min_h_lr, min_w_lr)
    batch_target_images = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)

    for idx, item in enumerate(batch):
        # 直接裁切到 min_h, min_w
        #x_lr_crop = item['input_lr'][:, :, :min_h_lr, :min_w_lr]
        input_crop   = item['input_hr'][:, :, :min_h_hr, :min_w_hr]
        label_crop  = item['target_img'][:, :, :min_h_hr, :min_w_hr]

        # 放到 batch
        #batch_x_lr[idx] = x_lr_crop
        batch_input[idx] = input_crop
        batch_target_images[idx] = label_crop

    return batch_input, batch_target_images

##### 构建训练集和测试集 ####
train_dataset = GuidedFilterDataset(
    input_dir = input_folder_train,
    label_dir = input_folder_train_label
)

test_dataset = GuidedFilterDataset(
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
    full_path = os.path.join(save_dir, "refine_filter_full_v2.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss
    }, full_path)
    print(f"Checkpoint saved: {full_path}")

    # 保存纯模型 (方便推理或部署)
    last_model_path = os.path.join(save_dir, "refine_filter_last_v2.pth")
    torch.save(model.state_dict(), last_model_path)
    print(f"Last model weights saved: {last_model_path}")

    # 如果是最佳模型，另存一份
    if is_best:
        best_model_path = os.path.join(save_dir, f"refine_filter_best_v2.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"*** Best model updated: {best_model_path} (Test MSE: {best_loss:.6f}) ***")

##### tensorboard 启动指令 #######
task_name = "convguided_filter_v16_refinev2"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/log_dir", f"{task_name}_{timestamp}")
writer = SummaryWriter(log_dir=log_dir)

######### psnr 计算 ###########
def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(255.0 * 255.0 / mse)
    return psnr, mse


device = torch.device('cuda:0')
model = EdgeAwareRefineNet(in_channels=1)
model = torch.nn.DataParallel(model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5) 
best_loss = float("inf")
save_dir = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/models"
epochs = 80

if __name__ == "__main__":
######## -- 模型训练 -- #############
    for epoch in tqdm(range(epochs), desc="Training Progress", dynamic_ncols=True):
        model.train()
        epoch_mse = 0
        epoch_psnr = 0
        for batch_idx, (batch_input, batch_label) in enumerate(train_loader):        
            #t0 = time.time()

            #batch_x_lr = crop_to_even(batch_x_lr)
            batch_input = crop_to_even(batch_input)
            batch_label = crop_to_even(batch_label)

            batch_input, batch_label = batch_input.to(device), batch_label.to(device)
         
            #print(batch_edge.device, batch_coarse.device, batch_label.device)
            #batch_x_lr = batch_x_lr - 128
            batch_input = batch_input - 128
            batch_label = batch_label - 128
            
            #t1 = time.time()
            # 前向
            output = model(batch_input)
            
            #output = model(batch_input)

            # if batch_idx % 10 == 0:
            #     save_debug_images( batch_x_lr, batch_input, output, batch_label, batch_idx)
            #     print('已经保存了，狗熊')
            
            loss = edge_sensitive_loss(output, batch_label, edge_weight=10.0)
            #loss = edge_position_loss(output, batch_label, grad_thresh=0.1)

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #t3 = time.time()

            # 计算 PSNR 和 MSE
            # if not torch.isfinite(output).all() or not torch.isfinite(batch_label).all():
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
            for test_input, test_label in test_loader:
                #test_x_lr = crop_to_even(test_x_lr)
                test_input = crop_to_even(test_input)
                test_label = crop_to_even(test_label)

                test_input, test_label = test_input.to(device), test_label.to(device)
                #test_x_lr = test_x_lr - 128
                test_input = test_input - 128
                test_label = test_label - 128

                test_output = model(test_input)
                #test_output = model(test_input)

                loss = edge_sensitive_loss(test_output, test_label, edge_weight=10.0)
                #loss = edge_position_loss(test_output, test_label, grad_thresh=0.1)

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


  









