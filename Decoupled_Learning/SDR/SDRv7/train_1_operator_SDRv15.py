import random
import numpy as np
import torch

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import cv2
import scipy.misc
import os
from skimage import io
from os import listdir
from os.path import join
from PIL import Image
import time
import glob
import tifffile

#torch.autograd.set_detect_anomaly(True)
# --- 1 Direct Intrinsic Network  ----
# 残差块（支持空洞卷积）
class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels=32, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)  # 残差连接

class DirectIntrinsicNet(nn.Module):
    def __init__(self):
        super().__init__()
        #self.pre_add = -128.0  # 对应 AddConstant(-115)

        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, 2, 1)  # 下采样 stride=2
        self.bn3 = nn.BatchNorm2d(32)

        # 堆叠的 dilated 残差块
        self.blocks = nn.Sequential(
            ResidualDilatedBlock(32, dilation=2),
            ResidualDilatedBlock(32, dilation=2),
            ResidualDilatedBlock(32, dilation=4),
            ResidualDilatedBlock(32, dilation=4),
            ResidualDilatedBlock(32, dilation=8),
            ResidualDilatedBlock(32, dilation=8),
            ResidualDilatedBlock(32, dilation=16),
            ResidualDilatedBlock(32, dilation=16),
            ResidualDilatedBlock(32, dilation=1),
            ResidualDilatedBlock(32, dilation=1),
        )

        # 上采样 (对应 SpatialFullConvolution)
        self.deconv = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv_final1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final2 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        #x = x + self.pre_add

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)

        x = self.blocks(x)

        x = F.relu(self.bn4(self.deconv(x)), inplace=True)
        x = F.relu(self.bn5(self.conv_final1(x)), inplace=True)
        x = torch.sigmoid(self.conv_final2(x)) ######### 输出做一个sigmoid 注意！！！
        return x
# --- 2. Guidance Network (18层, 只接收单通道输入) ---
class GuidanceNet(nn.Module):
    """ 完整网络，输入为灰度图+边缘图，输出单通道 """
    def __init__(self):
        super().__init__()
        #self.pre_add = -128.0  # 对应 AddConstant(-128)

        # 初始特征提取 (输入 2 通道)
        self.conv1 = nn.Conv2d(2, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 2, 1)  # stride=2 下采样
        self.bn3 = nn.BatchNorm2d(64)

        # 堆叠空洞卷积残差块
        self.blocks = nn.Sequential(
            ResidualDilatedBlock(64, dilation=2),
            ResidualDilatedBlock(64, dilation=2),
            ResidualDilatedBlock(64, dilation=4),
            ResidualDilatedBlock(64, dilation=4),
            ResidualDilatedBlock(64, dilation=8),
            ResidualDilatedBlock(64, dilation=8),
        )

        # 上采样 (反卷积)
        self.deconv = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(64)

        # 输出层
        self.conv_final1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv_final2 = nn.Conv2d(64, 1, 1)  # 单通道输出

    def forward(self, gray_img, edge_img):
        # 输入拼接 (gray 1通道 + edge 1通道 → 4通道)
        x = torch.cat([gray_img, edge_img], dim=1)  # (B,2,H,W)
        #x = x + self.pre_add  # 偏置处理

        # 初始卷积
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)

        # 残差 dilated 卷积
        x = self.blocks(x)

        # 上采样和输出
        x = F.relu(self.bn4(self.deconv(x)), inplace=True)
        x = F.relu(self.bn5(self.conv_final1(x)), inplace=True)
        x = self.conv_final2(x)

        # 最后加上 edge 分支 (残差), 0.5 表示恢复成原来的数据尺度

        out = x.clone() + edge_img.clone() + 0.5

        # 数值限制在 [0,1]
        out = torch.tanh(out)      # 限制在 [-1,1]
        out = (out + 1) / 2        # 映射到 [0,1]

        return out

# --- 3. Domain Filter ---
class DomainTransform(nn.Module):
    def __init__(self, num_iter=5, sigma_range=0.25, sigma_spatial=30, average=True):
        super().__init__()
        self.num_iter = num_iter
        self.sigma_range = sigma_range
        self.sigma_spatial = sigma_spatial
        self.average = average  # True: data+edge(2通道), False: RGB+edge(4通道)

    def forward(self, x):
        """
        x: [B, C, H, W]
            - average=True:  C=2  (data+edge)
            - average=False: C=4  (RGB+edge)
        返回:
            平滑后的 data
        """
        if self.average:
            data, edge = x[:, 0:1], x[:, 1:2]
        else:
            data, edge = x[:, 0:3], x[:, 3:4]

        out = data.clone()

        # 多次迭代平滑
        for k in range(1, self.num_iter + 1):
            # 递减的空间尺度
            sigma_Hk = self.sigma_spatial * (3 ** 0.5) * (2 ** (self.num_iter - k)) / ((4 ** self.num_iter - 1) ** 0.5)
            # 计算平滑系数 g
            g = torch.exp(- (1 + sigma_Hk) * self.sigma_range * edge)

            # 水平递归
            out = self.recursive_pass(out, g, dim=3)
            # 垂直递归
            out = self.recursive_pass(out, g, dim=2)

        return out

    def recursive_pass(self, x, g, dim):
        """
        双向递归滤波 (非 inplace 版本，避免梯度冲突)
        Y_i = (1 - g_i) * X_i + g_i * Y_{i-1}
        """
        # Forward pass
        x_f = x.clone()
        x_f_new = x_f.clone()
        for i in range(1, x.size(dim)):
            idx_prev = [slice(None)] * 4
            idx_curr = [slice(None)] * 4
            idx_prev[dim], idx_curr[dim] = i - 1, i
            new_val = (1 - g[tuple(idx_curr)]) * x_f[tuple(idx_curr)] + g[tuple(idx_curr)] * x_f[tuple(idx_prev)]
            x_f_new[tuple(idx_curr)] = new_val
        x_f = x_f_new  # 替换成新的，不改原张量

        # Backward pass
        x_b = x_f.clone()
        x_b_new = x_b.clone()
        for i in range(x.size(dim) - 2, -1, -1):
            idx_next = [slice(None)] * 4
            idx_curr = [slice(None)] * 4
            idx_next[dim], idx_curr[dim] = i + 1, i
            new_val = (1 - g[tuple(idx_curr)]) * x_b[tuple(idx_curr)] + g[tuple(idx_curr)] * x_b[tuple(idx_next)]
            x_b_new[tuple(idx_curr)] = new_val
        x_b = x_b_new

        return x_b

# --- 4. 整体网络结构 ---
class EdgePreservingNet(nn.Module):
    def __init__(self, num_iter=5, sigma_range=0.25, sigma_spatial=30):
        super(EdgePreservingNet, self).__init__()
        self.guidance_net = GuidanceNet()
        self.intrinsic_net = DirectIntrinsicNet()
        self.domain_transform = DomainTransform(
            num_iter=num_iter,
            sigma_range=sigma_range,
            sigma_spatial=sigma_spatial,
            average=True  # 这里用 gray+edge 模式
        )

    def forward(self, guidance_input, input_img):
        """-
        guidance_input: [B, 1, H, W] (引导图输入, 可能是灰度)
        input_img: [B, 1, H, W] (IntrinsicNet 输入图像)
        返回:
            R_coarse: [B, 3, H, W] 反射率预测
            edge_map: [B, 1, H, W] 引导边缘图
            R_final:  [B, 3, H, W] 经 DomainTransform 平滑后的反射率
        """
        # 1. 粗预测base层 (1通道)
        B_coarse = self.intrinsic_net(input_img)  # [B,3,H,W]

        # 2. 引导边缘图 (GuidanceNet 输出)
        input_edge = guidance_input[:, 1:2, :, :].clone()
        edge_map = self.guidance_net(input_img, input_edge)  # [B,1,H,W] 或 [B,2,H,W]，按你实际网络定义

        # 确保 edge_map 最终是单通道
        if edge_map.shape[1] > 1:
            # 如果 GuidanceNet 输出多通道, 这里取第一通道或者均值
            edge_map = edge_map.mean(dim=1, keepdim=True)  # [B,1,H,W]

        # 3. 拼接为 DomainTransform 的输入: RGB(3) + edge(1)
        dt_input = torch.cat([B_coarse, edge_map], dim=1)  # [B,4,H,W]

        # 4. 经过 DomainTransform 平滑
        B_final = self.domain_transform(dt_input)  # [B,3,H,W]

        return B_coarse, edge_map, B_final

'''
class ArbitraryImageFolder(data.Dataset):

    def __init__(self, guidance_input_paths, guidance_edge_paths, net_input_paths, guidance_label_paths, net_label_paths, params, types, hws):
        self.guidance_input_paths = guidance_input_paths
        self.guidance_edge_paths = guidance_edge_paths
        self.net_input_paths = net_input_paths
        self.guidance_label_paths = guidance_label_paths
        self.net_label_paths = net_label_paths

        self.params = params
        self.types = types
        self.hws = hws
        self.size = len(net_input_paths)  # 以主输入长度为基准

    def __getitem__(self, index):
        # 路径
        guidance_input_path = self.guidance_input_paths[index]
        guidance_edge_path = self.guidance_edge_paths[index]
        net_input_path = self.net_input_paths[index]
        guidance_label_path = self.guidance_label_paths[index]
        net_label_path = self.net_label_paths[index]

        para = self.params[index]
        typ = self.types[index]
        img_h, img_w = self.hws[index]

        # 读取图像
        # 1. guidance net 输入
        guidance_input = cv2.imread(guidance_input_path, cv2.IMREAD_GRAYSCALE)
        guidance_input = guidance_input[:, :, np.newaxis]
        guidance_input = guidance_input.transpose((2, 0, 1))
        guidance_input = torch.from_numpy(guidance_input).float()

        # 2. guidance net edge label
        guidance_edge = cv2.imread(guidance_edge_path, cv2.IMREAD_GRAYSCALE)
        guidance_edge = guidance_edge[:, :, np.newaxis]
        guidance_edge = guidance_edge.transpose((2, 0, 1))
        guidance_edge = torch.from_numpy(guidance_edge).float()

        # 3. main net 输入
        net_input = cv2.imread(net_input_path, cv2.IMREAD_GRAYSCALE)
        net_input = net_input[:, :, np.newaxis]
        net_input = net_input.transpose((2, 0, 1))
        net_input = torch.from_numpy(net_input).float()

        # 4. guidance net label
        guidance_label = cv2.imread(guidance_label_path, cv2.IMREAD_GRAYSCALE)
        guidance_label = guidance_label[:, :, np.newaxis]
        guidance_label = guidance_label.transpose((2, 0, 1))
        guidance_label = torch.from_numpy(guidance_label).float()

        # 5. main net label
        net_label = cv2.imread(net_label_path, cv2.IMREAD_GRAYSCALE)
        net_label = net_label[:, :, np.newaxis]
        net_label = net_label.transpose((2, 0, 1))
        net_label = torch.from_numpy(net_label).float()

        return {
            'guidance_input': guidance_input,           # guidance net的输入
            'guidance_edge': guidance_edge,             # guidance net的edge label
            'net_input': net_input,                     # main net输入
            'guidance_label': guidance_label,           # guidance net label
            'net_label': net_label,                     # main net label
            'input_param': para,
            'input_type': typ,
            'input_height': img_h,
            'input_width': img_w
        }

    def __len__(self):
        return self.size
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

batchsize = 16
num_workers = 16

print("***********************************************")
########## TRAIN Folder ##########
guidance_input_folder = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRinput'
guidance_edge_input_folder = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/guided_edge_train_input'
guidance_label_folder = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/guided_edge_train_label'

net_input_folder = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRinput'
net_label_folder = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRinput_label'
########## TEST Folder ###########
guidance_input_folder_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRtest'
guidance_edge_input_folder_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/guided_edge_test_input'
guidance_label_folder_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/guided_edge_test_label'

net_input_folder_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRtest'
net_label_folder_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRtest_label'

########### 数据加载 ##############
class GuidedFilterDataset(Dataset):
    def __init__(self, input_dir, guidance_edge_input_dir, guidance_edge_label_dir, net_label_dir):
        self.input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        self.guidance_edge_input_files = sorted([os.path.join(guidance_edge_input_dir, f) for f in os.listdir(guidance_edge_input_dir)])
        self.guidance_edge_label_files = sorted([os.path.join(guidance_edge_label_dir, f) for f in os.listdir(guidance_edge_label_dir)])
        self.net_label_files = sorted([os.path.join( net_label_dir, f) for f in os.listdir( net_label_dir)])

        print(f"input: {len(self.input_files)}, edge_input: {len(self.guidance_edge_input_files)}, edge-label: {len(self.guidance_edge_label_files)}, label: {len(self.net_label_files)}")

    def __len__(self):
        return min(len(self.input_files), len(self.guidance_edge_input_files),
                    len(self.guidance_edge_label_files), len(self.net_label_files))

    def __getitem__(self, idx):
        input       = cv2.imread(self.input_files[idx], cv2.IMREAD_GRAYSCALE).astype('float32')
        edge_input  = cv2.imread(self.guidance_edge_input_files[idx], cv2.IMREAD_GRAYSCALE).astype('float32')
        edge_label  = cv2.imread(self.guidance_edge_label_files[idx], cv2.IMREAD_GRAYSCALE).astype('float32')
        input_label = cv2.imread(self.net_label_files[idx], cv2.IMREAD_GRAYSCALE).astype('float32')

        input_h, input_w = input.shape
        #input_lr_h, input_lr_w = x_lr.shape

        return {
            'input': torch.from_numpy(input).unsqueeze(0).unsqueeze(0),  # [1,1,H,W]
            'edge_input': torch.from_numpy(edge_input).unsqueeze(0).unsqueeze(0),
            'edge_label': torch.from_numpy(edge_label).unsqueeze(0).unsqueeze(0),
            'input_label': torch.from_numpy(input_label).unsqueeze(0).unsqueeze(0),

            'input_height': input_h,
            'input_width': input_w
        }
    
def var_custom_collate(batch):
    #min_h, min_w = 10000, 10000

    min_h = min([item['input_height'] for item in batch])
    min_w = min([item['input_width'] for item in batch])   

    batch_coarse_input = torch.zeros(len(batch), 1, min_h, min_w)
    batch_guidance_input   = torch.zeros(len(batch), 2, min_h, min_w)
    batch_target_images = torch.zeros(len(batch), 1, min_h, min_w)
    batch_guidance_label = torch.zeros(len(batch),1, min_h, min_w)

    for idx, item in enumerate(batch):
        input       = item['input'][:, :, :min_h, :min_w]
        input_label = item['input_label'][:, :, :min_h, :min_w]
        edge_input  = item['edge_input'][:, :, :min_h, :min_w]
        edge_label  = item['edge_label'][:, :, :min_h, :min_w]

        # 放到 batch
        batch_coarse_input[idx] = input
        batch_guidance_input[idx,0:1] = input
        batch_guidance_input[idx, 1] = edge_input
        batch_guidance_label[idx] = edge_label
        batch_target_images[idx] = input_label

    return batch_coarse_input, batch_guidance_input,batch_guidance_label, batch_target_images

train_dataset = GuidedFilterDataset(
    input_dir = net_input_folder,
    guidance_edge_input_dir = guidance_edge_input_folder,
    guidance_edge_label_dir = guidance_label_folder,
    net_label_dir = net_label_folder
)

test_dataset = GuidedFilterDataset(
    input_dir = net_input_folder_test,
    guidance_edge_input_dir = guidance_edge_input_folder_test,
    guidance_edge_label_dir = guidance_label_folder_test,
    net_label_dir = net_label_folder_test
)

train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=num_workers,
                          collate_fn=var_custom_collate, pin_memory=True, prefetch_factor=4, drop_last=True)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1,
                          collate_fn=var_custom_collate, pin_memory=True, drop_last=False)


##### tensorboard 启动指令 #######
#writer = SummaryWriter(log_dir="/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/log_dir")

import os, datetime
from tqdm import tqdm
task_name = "domain_filter_v15_2"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/log_dir", f"{task_name}_{timestamp}")

writer = SummaryWriter(log_dir=log_dir)

# --- 在每个 epoch 结束后保存 ---
def save_checkpoint(epoch, best_loss, is_best=False):
    # 保存完整状态，方便续训
    full_path = os.path.join(save_dir, "domain_filter_v15_2.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss
    }, full_path)
    print(f"Checkpoint saved: {full_path}")

    # 保存纯模型 (方便推理或部署)
    last_model_path = os.path.join(save_dir, "domain_filter_v15_2_last.pth")
    torch.save(model.state_dict(), last_model_path)
    print(f"Last model weights saved: {last_model_path}")

    # 如果是最佳模型，另存一份
    if is_best:
        best_model_path = os.path.join(save_dir, f"domain_filter_v15_2_best.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"*** Best model updated: {best_model_path} (Test MSE: {best_loss:.6f}) ***")

######### psnr 计算 ###########
def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1.0 / mse)
    return psnr, mse

######### 长款边裁剪 ######
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

######### loss 和梯度计算 ############
def gradient_x(img):
    return img[:, :, :, 1:] - img[:, :, :, :-1]

def gradient_y(img):
    return img[:, :, 1:, :] - img[:, :, :-1, :]

def ldi_loss(B_coarse, batch_coarse_input, batch_target_images, lambda1=1.0, lambda2=1.0, auto_balance=False, model=None):
    batch_coarse_input_restored = batch_coarse_input + 0.5 # 恢复到 0-1
    D_coarse = batch_coarse_input_restored - B_coarse
    D_target = batch_coarse_input_restored - batch_target_images

    # 先把 target 从 0-1 转换回 -128~128
    target_rescaled = batch_target_images
    D_target_rescaled = D_target  # D_target 已经和 batch_coarse_input 一样，不用变

    # L2 reconstruction terms (R' vs R*, S vs S*)
    recon_B = torch.mean((B_coarse - target_rescaled) ** 2)
    recon_D = torch.mean((D_coarse - D_target_rescaled) ** 2)

    # Gradient terms for B (R' vs R*)
    grad_Bx = torch.mean((gradient_x(B_coarse) - gradient_x(target_rescaled)) ** 2)
    grad_By = torch.mean((gradient_y(B_coarse) - gradient_y(target_rescaled)) ** 2)

    # Gradient terms for D (S vs S*)
    grad_Dx = torch.mean((gradient_x(D_coarse) - gradient_x(D_target_rescaled)) ** 2)
    grad_Dy = torch.mean((gradient_y(D_coarse) - gradient_y(D_target_rescaled)) ** 2)

    # Combine all terms 原始做法, 直接相加，手动调权重(v15 & v15_1)
    #loss = lambda2 * (recon_B + recon_D) + lambda1 * (grad_Bx + grad_By + grad_Dx + grad_Dy)

    ## v15_2 的权重调整。通过gradnorm进行
    grad_loss = grad_Bx + grad_By + grad_Dx + grad_Dy
    recon_loss = recon_B + recon_D

    if auto_balance and model is not None:
        g_recon = compute_grad_norm(recon_loss, model)
        g_grad = compute_grad_norm(grad_loss, model)
        # 反比缩放，归一化到1
        s_recon = 1.0 / max(g_recon, 1e-8)
        s_grad = 1.0 / max(g_grad, 1e-8)
        s_sum = s_recon + s_grad
        lambda2 = s_recon / s_sum
        lambda1 = s_grad / s_sum    

    return lambda2 * recon_loss + lambda1 * grad_loss

########## 梯度归一化，用于调整loss的权重 ############
def compute_grad_norm(loss, model):
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, allow_unused=True)
    total_norm = 0
    for g in grads:
        if g is not None:
            total_norm += g.norm().item()
    return total_norm

############## 模型初始化及前向传播 ################
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
model = EdgePreservingNet()
model = torch.nn.DataParallel(model.cuda())
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5) 
best_psnr = float("inf")
save_dir = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/models"

###### 重新加载训练模型 ###########
resume_path = os.path.join(save_dir, "domain_filter_v15_2.pth")
if os.path.exists(resume_path):
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']  # 从上次的 epoch 接着跑
    best_loss = checkpoint['best_loss']
    print(f"=> Resumed from checkpoint (epoch {start_epoch})")

else:
    print("=> No checkpoint found, training from scratch.")


epochs = 80
criterion = torch.nn.MSELoss(reduction='mean')

if __name__ == "__main__":
    for epoch in tqdm(range(start_epoch, epochs), desc="Training Progress", dynamic_ncols=True):
        epoch_start = time.time()
        model.train()
        epoch_data_time, epoch_forward_time, epoch_backward_time = 0, 0, 0

        epoch_mse_c, epoch_psnr_c = 0.0, 0.0
        epoch_mse_g, epoch_psnr_g = 0.0, 0.0
        epoch_mse_f, epoch_psnr_f = 0.0, 0.0
        epoch_loss_total = 0.0
########################### 训练阶段 #################################################
        for batch_idx, (batch_coarse_input, batch_guidance_input,batch_guidance_label, batch_target_images) in enumerate(train_loader):  
            t0 = time.time()
            batch_coarse_input, batch_guidance_input,batch_guidance_label, batch_target_images = batch_coarse_input.cuda(), batch_guidance_input.cuda(), batch_guidance_label.cuda(), batch_target_images.cuda()
            epoch_data_time += time.time() - t0

            #### 预处理 ####
            t1 = time.time()
            batch_coarse_input = (batch_coarse_input - 128) / 255 # -0.5 ~ 0.5
            batch_guidance_input = (batch_guidance_input -128) / 255 # -0.5 ~ 0.5

            batch_target_images  = batch_target_images / 255
            batch_guidance_label = batch_guidance_label / 255

            # 先裁剪到偶数尺寸
            batch_coarse_input = crop_to_even(batch_coarse_input)
            batch_guidance_input = crop_to_even(batch_guidance_input)
            batch_target_images = crop_to_even(batch_target_images)
            batch_guidance_label = crop_to_even(batch_guidance_label)
            epoch_data_time += time.time() - t1

            ##########前向 #######
            t2 = time.time()
            B_coarse, edge_map, B_final = model(batch_guidance_input, batch_coarse_input) # 0-1 scale
            epoch_forward_time += time.time() - t2
            # D_coarse = batch_coarse_input - B_coarse
            # D_target = batch_coarse_input - batch_target_images

            ##### LOSS 计算 #######
            loss_f = criterion(B_final, batch_target_images) # 0-1 scale
            loss_g = criterion(edge_map, batch_guidance_label) # 0-1 scale
            #loss_di = ldi_loss(B_coarse, batch_coarse_input, batch_target_images, lambda1=0.35, lambda2=0.2)
            ##### v15_2 ####
            loss_di = ldi_loss(B_coarse, batch_coarse_input, batch_target_images, auto_balance=True, model=model)

############### 不同的 loss 调整方案 ################
            ######### v15 ###############
            #loss = loss_di + 0.35*loss_g + 0.2*loss_f
            ######### v15_1 #############
            #loss = loss_di + loss_g + loss_f
            ######### v15_2 #############
            losses = [loss_di, loss_g, loss_f]
            loss_vals = [l.detach().item() for l in losses]
            inv_weights = [1.0 / (lv + 1e-8) for lv in loss_vals]
            sum_inv = sum(inv_weights)
            weights = [iw / sum_inv for iw in inv_weights]
            loss = sum(w * l for w, l in zip(weights, losses))

            # 每40个batch输出一次当前权重
            if batch_idx % 40 == 0:
                print(f"Dynamic Weights (Loss-based) -> DI: {weights[0]:.3f}, Edge: {weights[1]:.3f}, Final: {weights[2]:.3f}")

            ###### 反向传播 #####
            t3 = time.time()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_backward_time += time.time() - t3

            psnr_coarse, mse_coarse = calc_psnr(B_coarse.detach(), batch_target_images)
            psnr_guidance, mse_guidance = calc_psnr(edge_map.detach(), batch_guidance_label)
            psnr_final, mse_final = calc_psnr(B_final.detach(), batch_target_images)

            # 7. TensorBoard 记录
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/mse_coarse', mse_coarse, global_step)
            writer.add_scalar('train/psnr_coarse', psnr_coarse, global_step)

            writer.add_scalar('train/mse_edge', mse_guidance, global_step)
            writer.add_scalar('train/psnr_edge', psnr_guidance, global_step)

            writer.add_scalar('train/mse_final', mse_final, global_step)
            writer.add_scalar('train/psnr_final', psnr_final, global_step)

            writer.add_scalar('train/total_loss', loss.item(), global_step)

            # 8. 控制台输出
            if batch_idx % 20 == 0:
                print(f"Iter [{batch_idx}/{len(train_loader)}] "
                    f"Coarse PSNR: {psnr_coarse:.2f}, Final PSNR: {psnr_final:.2f}, Edge PSNR: {psnr_guidance:.2f}, "
                    f"Loss: {loss.item():.4f}")         

            # 9. 累积 epoch 级别统计 (如果你需要 epoch 平均指标)
            epoch_mse_c += mse_coarse
            epoch_psnr_c += psnr_coarse
            epoch_mse_g += mse_guidance
            epoch_psnr_g += psnr_guidance
            epoch_mse_f += mse_final
            epoch_psnr_f += psnr_final
            epoch_loss_total += loss

        print(f"Epoch {epoch+1}: "
          f"Data {epoch_data_time:.2f}s, "
          f"Forward {epoch_forward_time:.2f}s, "
          f"Backward {epoch_backward_time:.2f}s, "
          f"Total {time.time() - epoch_start:.2f}s")
        
        # 每个 epoch 结束后
        avg_epoch_mse_c = epoch_mse_c / len(train_loader)
        avg_epoch_psnr_c = epoch_psnr_c / len(train_loader)

        avg_epoch_mse_g = epoch_mse_g / len(train_loader)
        avg_epoch_psnr_g = epoch_psnr_g / len(train_loader)

        avg_epoch_mse_f = epoch_mse_f / len(train_loader)
        avg_epoch_psnr_f = epoch_psnr_f / len(train_loader)

        avg_epoch_loss_total = epoch_loss_total / len(train_loader)

        # 打印本 epoch 的平均指标
        print(f"Epoch {epoch+1}/{epochs} "
            f"- Coarse MSE: {avg_epoch_mse_c:.6f}, PSNR: {avg_epoch_psnr_c:.2f} dB "
            f"- Edge MSE: {avg_epoch_mse_g:.6f}, PSNR: {avg_epoch_psnr_g:.2f} dB "
            f"- Final MSE: {avg_epoch_mse_f:.6f}, PSNR: {avg_epoch_psnr_f:.2f} dB "
            f"- Loss: {avg_epoch_loss_total:.6f}")

        # 写入 TensorBoard (按 epoch 记录)
        writer.add_scalar('epoch/mse_coarse', avg_epoch_mse_c, epoch)
        writer.add_scalar('epoch/psnr_coarse', avg_epoch_psnr_c, epoch)

        writer.add_scalar('epoch/mse_edge', avg_epoch_mse_g, epoch)
        writer.add_scalar('epoch/psnr_edge', avg_epoch_psnr_g, epoch)

        writer.add_scalar('epoch/mse_final', avg_epoch_mse_f, epoch)
        writer.add_scalar('epoch/psnr_final', avg_epoch_psnr_f, epoch)

        writer.add_scalar('epoch/loss_total', avg_epoch_loss_total, epoch)
        scheduler.step()
        
#################### 测试集验证 ###########################
            
        model.eval()
        val_mse_c, val_psnr_c = 0.0, 0.0
        val_mse_g, val_psnr_g = 0.0, 0.0
        val_mse_f, val_psnr_f = 0.0, 0.0
        val_loss_total, val_psnr_total = 0.0, 0.0

        with torch.no_grad():
            for batch_coarse_input, batch_guidance_input, batch_guidance_label, batch_target_images in test_loader:
                batch_coarse_input = batch_coarse_input.cuda()
                batch_guidance_input = batch_guidance_input.cuda()
                batch_guidance_label = batch_guidance_label.cuda()
                batch_target_images = batch_target_images.cuda()

                # 预处理 (与训练一致)
                batch_coarse_input = (batch_coarse_input - 128) / 255
                batch_guidance_input = (batch_guidance_input - 128) / 255
                batch_target_images = batch_target_images / 255.0
                batch_guidance_label = batch_guidance_label / 255.0

                batch_coarse_input = crop_to_even(batch_coarse_input)
                batch_guidance_input = crop_to_even(batch_guidance_input)
                batch_target_images = crop_to_even(batch_target_images)
                batch_guidance_label = crop_to_even(batch_guidance_label)

                B_coarse, edge_map, B_final = model(batch_guidance_input, batch_coarse_input) ### 输出的中间图和结果图（B_final）

                # Loss 计算
                loss_f = criterion(B_final, batch_target_images)
                loss_g = criterion(edge_map, batch_guidance_label)
                #loss_di = ldi_loss(B_coarse, batch_coarse_input, batch_target_images, lambda1=0.35, lambda2=0.2)
                ######## v15 ############
                #loss = loss_di + 0.35 * loss_g + 0.2 * loss_f
                ######## v15_1 ############
                #loss = loss_di + loss_g + loss_f
                ######## v15_2 ###############
                # losses = [loss_di, loss_g, loss_f]
                # grads = [compute_grad_norm(l, model) for l in losses]
                # scales = [1.0 / max(g, 1e-8) for g in grads]
                # sum_s = sum(scales)
                # weights = [s / sum_s for s in scales]
                # loss = sum(w * l for w, l in zip(weights, losses))

                psnr_coarse, mse_coarse = calc_psnr(B_coarse, batch_target_images)
                psnr_edge, mse_edge = calc_psnr(edge_map, batch_guidance_label)
                psnr_final, mse_final = calc_psnr(B_final, batch_target_images)

                val_mse_c += mse_coarse
                val_psnr_c += psnr_coarse
                val_mse_g += mse_edge
                val_psnr_g += psnr_edge
                val_mse_f += mse_final
                val_psnr_f += psnr_final
                #val_loss_total += loss.item()

        num_val_batches = len(test_loader)
        avg_val_mse_c = val_mse_c / num_val_batches
        avg_val_psnr_c = val_psnr_c / num_val_batches

        avg_val_mse_g = val_mse_g / num_val_batches
        avg_val_psnr_g = val_psnr_g / num_val_batches

        avg_val_mse_f = val_mse_f / num_val_batches
        avg_val_psnr_f = val_psnr_f / num_val_batches

        #avg_val_loss_total = val_loss_total / num_val_batches


        print(f"[Validation Epoch {epoch+1}] "
            f"Coarse PSNR: {avg_val_psnr_c:.2f} dB, Final PSNR: {avg_val_psnr_f:.2f} dB, Edge PSNR: {avg_val_psnr_g:.2f} dB ")
        
        # ======= 写入 TensorBoard =======
        writer.add_scalar('val/mse_coarse', avg_val_mse_c, epoch)
        writer.add_scalar('val/psnr_coarse', avg_val_psnr_c, epoch)

        writer.add_scalar('val/mse_edge', avg_val_mse_g, epoch)
        writer.add_scalar('val/psnr_edge', avg_val_psnr_g, epoch)

        writer.add_scalar('val/mse_final', avg_val_mse_f, epoch)
        writer.add_scalar('val/psnr_final', avg_val_psnr_f, epoch)

        #writer.add_scalar('val/loss_total', avg_val_loss_total, epoch)

        # 判断是否最佳模型
        is_best = avg_val_psnr_f < best_psnr
        if is_best:
            best_psnr = avg_val_psnr_f

        # 保存模型
        save_checkpoint(epoch, best_psnr, is_best=is_best)
        # 关闭 TensorBoard writer
    writer.close()













'''

trainSetSize, testSetSize = len(train_net_input_paths), len(test_net_input_paths)
epochSize = 279
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EdgePreservingNet()
model = torch.nn.DataParallel(model.cuda())
model.train()

criterion = torch.nn.MSELoss(size_average=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_dataset = ArbitraryImageFolder(train_guidance_input_paths, train_guidance_input_edge_paths, train_net_input_paths, train_gudiance_label_paths, train_net_label_paths,
                            train_params, train_type, train_hws)
train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, collate_fn=var_custom_collate,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

test_dataset = ArbitraryImageFolder(test_guidance_input_paths, test_guidance_input_edge_paths, test_net_input_paths, test_gudiance_label_paths, test_net_label_paths,
                            test_params, test_type, test_hws)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, collate_fn=var_custom_collate,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

total_start_time = time.time()
iter = 0
iterTest = 0
switch_epoch = 100
writer = SummaryWriter('vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/checkpoints/wls_log')
for p in range(epoches):
    train_loss = 0
    test_loss = 0
    train_count = 0
    test_count = 0
    start_time = time.time()
    if p == 80:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for curr_iter, batch_data in enumerate(train_dataloader):
        iter += 1
        guidance_input_Tensor, net_input_Tensor, inputParaTensor, guidance_label_Tensor, net_label_Tensor = batch_data
        
        guidance_input_Tensor = guidance_input_Tensor - 128
        net_input_Tensor = net_input_Tensor - 128

        guidance_input_Tensor_v = Variable(guidance_input_Tensor.cuda())
        net_input_Tensor_v = Variable(net_input_Tensor.cuda())
        inputParaTensor_v = Variable(inputParaTensor.cuda())
        guidance_label_Tensor_v = Variable(guidance_label_Tensor.cuda())
        net_label_Tensor_v = Variable(net_label_Tensor.cuda())

        optimizer.zero_grad()
        R_coarse, edge_map = model([inputParaTensor_v, guidance_input_Tensor_v], net_input_Tensor_v)

        min_h = min(edge_map.shape[2], guidance_label_Tensor_v.shape[2])
        min_w = min(edge_map.shape[3], guidance_label_Tensor_v.shape[3])
        edge_map = edge_map[:, :, :min_h, :min_w]
        guidance_label_Tensor_v = guidance_label_Tensor_v[:, :, :min_h, :min_w]

        min_h = min(R_coarse.shape[2], net_label_Tensor_v.shape[2])
        min_w = min(R_coarse.shape[3], net_label_Tensor_v.shape[3])
        R_coarse = R_coarse[:, :, :min_h, :min_w]
        net_label_Tensor_v = net_label_Tensor_v[:, :, :min_h, :min_w]
        ########## 计算得到edge map，R_coarse  并计算相关loss ###########
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        edge_map = edge_map.cuda()
        R_coarse = R_coarse.cuda()
        loss_intrinsic = criterion(R_coarse, net_label_Tensor_v)
        loss_guidance = criterion(edge_map, guidance_label_Tensor_v)

        ########### 将获得的edge map 和 R_coarse 输入 domain filter 获得最终的smoothed 结果 ###########
        min_h = min(edge_map.shape[2], R_coarse.shape[2])
        min_w = min(edge_map.shape[3], R_coarse.shape[3])
        edge_map = edge_map[:, :, :min_h, :min_w]
        R_coarse = R_coarse[:, :, :min_h, :min_w]
        smoothed = domain_filter(R_coarse, edge_map, eps=0.01)

        ########### 计算loss ###########
        min_h = min(smoothed.shape[2], net_label_Tensor_v.shape[2])
        min_w = min(smoothed.shape[3], net_label_Tensor_v.shape[3])
        smoothed = smoothed[:, :, :min_h, :min_w]
        net_label_Tensor_v = net_label_Tensor_v[:, :, :min_h, :min_w]
        loss_net = criterion(smoothed, net_label_Tensor_v)
        # loss_net = loss_net * 0.01
        # loss_guidance = loss_guidance * 0.4
        loss =  loss_guidance + loss_net + loss_intrinsic
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_count += 1

        writer.add_scalar('Loss/train_batch', loss.item(), iter)
        writer.add_scalar('Loss/train_batch_guidance', loss_guidance.item(), iter)
        writer.add_scalar('Loss/train_batch_net', loss_net.item(), iter)
        writer.add_scalar('Loss/train_batch_intrinsic', loss_intrinsic.item(), iter)

        print('Iter: %d | Total loss: %.4f | Guidance loss: %.4f | Net loss: %.4f | Intrinsic loss: %.4f'  % (
            iter, loss.item(), loss_guidance.item(), loss_net.item(), loss_intrinsic.item()
        ))
        if iter % int(epochSize/batchsize) == 0:
            break
    writer.add_scalar('Loss/train_epoch_avg', train_loss / train_count, p)
    for curr_iter, batch_data in enumerate(test_dataloader):
        iterTest += 1
        guidance_input_Tensor, net_input_Tensor, inputParaTensor, guidance_label_Tensor, net_label_Tensor = batch_data
        
        guidance_input_Tensor = guidance_input_Tensor - 128
        net_input_Tensor = net_input_Tensor - 128

        guidance_input_Tensor_v = Variable(guidance_input_Tensor.cuda())
        net_input_Tensor_v = Variable(net_input_Tensor.cuda())
        inputParaTensor_v = Variable(inputParaTensor.cuda())
        guidance_label_Tensor_v = Variable(guidance_label_Tensor.cuda())
        net_label_Tensor_v = Variable(net_label_Tensor.cuda())

        optimizer.zero_grad()
        R_coarse, edge_map = model([inputParaTensor_v, guidance_input_Tensor_v], net_input_Tensor_v)

        min_h = min(edge_map.shape[2], guidance_label_Tensor_v.shape[2])
        min_w = min(edge_map.shape[3], guidance_label_Tensor_v.shape[3])
        edge_map = edge_map[:, :, :min_h, :min_w]
        guidance_label_Tensor_v = guidance_label_Tensor_v[:, :, :min_h, :min_w]

        min_h = min(R_coarse.shape[2], net_label_Tensor_v.shape[2])
        min_w = min(R_coarse.shape[3], net_label_Tensor_v.shape[3])
        R_coarse = R_coarse[:, :, :min_h, :min_w]
        net_label_Tensor_v = net_label_Tensor_v[:, :, :min_h, :min_w]
################### 计算网络出来的两个loss 
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        edge_map = edge_map.cuda()
        R_coarse = R_coarse.cuda()
        loss_intrinsic = criterion(R_coarse, net_label_Tensor_v)
        loss_guidance = criterion(edge_map, guidance_label_Tensor_v)

        min_h = min(edge_map.shape[2], R_coarse.shape[2])
        min_w = min(edge_map.shape[3], R_coarse.shape[3])
        edge_map = edge_map[:, :, :min_h, :min_w]
        R_coarse = R_coarse[:, :, :min_h, :min_w]
        smoothed = domain_filter(R_coarse, edge_map, eps=0.01)

        min_h = min(smoothed.shape[2], net_label_Tensor_v.shape[2])
        min_w = min(smoothed.shape[3], net_label_Tensor_v.shape[3])
        smoothed = smoothed[:, :, :min_h, :min_w]
        net_label_Tensor_v = net_label_Tensor_v[:, :, :min_h, :min_w]
        loss_net = criterion(smoothed, net_label_Tensor_v)
################## 计算总loss ####
        loss =  loss_guidance + loss_net + loss_intrinsic

        test_loss += loss.item()
        test_count += 1

        if iterTest % int(epochSize/(batchsize*20)) == 0:
            break

    train_loss = train_loss / train_count
    test_loss = test_loss / test_count
    elapsed = time.time() - start_time
    print('Epoch: %d average train loss: %.4f'%(p+1,train_loss))
    print('Epoch: %d average test  loss: %.4f'%(p+1,test_loss))
    print('Epoch: %d time elapsed: %.2f hours'%(p+1,elapsed/3600))
    if (p + 1) % 100 == 0 :
        torch.save(
            model.module.state_dict(),
            f'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/models/model_{name}.net'
        )
total_elapsed = time.time() - total_start_time
writer.close()
print('Total time elapsed: %.2f days'%(total_elapsed/(3600*24)))

'''