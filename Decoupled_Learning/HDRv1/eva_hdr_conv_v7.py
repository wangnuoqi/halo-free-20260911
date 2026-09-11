import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from torch.nn import functional as F

import os, datetime

from tqdm import tqdm
from guided_filter_pytorch.box_filter import BoxFilter
#from guided_filter_pytorch.guided_filter import FastGuidedFilter
#from guided_filter_pytorch.guided_filter import GuidedFilter
#from guided_filter_pytorch.guided_filter import ConvGuidedFilter
import cv2
import numpy as np
import time
import math
import random
import imageio

# from train_convguided_v6 import DeepGuidedFilter
# from train_convguided_v6 import ConvGuidedFilter
# from train_convguided_v6 import build_lr_net

class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()
        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))
        self.bn  = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)

def weights_init_residual(m):
    classname = m.__class__.__name__
    
    # 1. 处理卷积层
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            m.bias.data.zero_()
            
    # 2. 处理标准的 BatchNorm2d
    elif classname.find('BatchNorm2d') != -1:
        if m.weight is not None:
            nn.init.constant_(m.weight.data, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

    # 3. 单独处理 AdaptiveNorm
    elif classname.find('AdaptiveNorm') != -1:
        # AdaptiveNorm 的目标是初始状态接近 Identity (恒等映射)
        # 所以我们希望：Output = 1.0 * x + 0.0 * BN(x)
        # 虽然你的类定义里已经默认初始化为 1 和 0 了，但为了保险起见，这里显式再设一次
        if hasattr(m, 'w_0'):
            nn.init.constant_(m.w_0, 1.0)
        if hasattr(m, 'w_1'):
            nn.init.constant_(m.w_1, 0.0)

class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1, norm=nn.BatchNorm2d):
        super(ConvGuidedFilter, self).__init__()
        print("Loaded ConvGuidedFilter from:", __file__)
        self.box_filter = nn.Conv2d(1, 1, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=1)
        self.conv_a = nn.Sequential(nn.Conv2d(2, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 1, kernel_size=1, bias=False))
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0))
        ## mean_x
        mean_x = self.box_filter(x_lr)/N
        ## mean_y
        mean_y = self.box_filter(y_lr)/N
        ## cov_xy
        cov_xy = self.box_filter(x_lr * y_lr)/N - mean_x * mean_y
        ## var_x
        var_x  = self.box_filter(x_lr * x_lr)/N - mean_x * mean_x

        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A * x_hr + mean_b

def build_lr_net(norm=AdaptiveNorm, layer=5):
    # ... (前面的层定义保持不变) ...
    layers = [
        nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
    ]

    for l in range(1, layer):
        layers += [nn.Conv2d(24,  24, kernel_size=3, stride=1, padding=2**l,  dilation=2**l,  bias=False),
                   norm(24),
                   nn.LeakyReLU(0.2, inplace=True)]
        
    layers += [
        nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),

        # 最后一层：输出 Residual Map (1 channel)
        nn.Conv2d(24, 1, kernel_size=1, stride=1, padding=0, dilation=1)
    ]

    net = nn.Sequential(*layers)

    # === 修改这里：使用标准初始化，而不是 Identity 初始化 ===
    net.apply(weights_init_residual) 
    
    # 这是一个小技巧：显式地将最后一层的权重缩小
    # 这样网络一开始预测的残差几乎为0，也就是 Output ≈ Input
    # 这对去噪/滤波任务是极好的起点
    net[-1].weight.data.mul_(0.1) 
    if net[-1].bias is not None:
        net[-1].bias.data.zero_()

    return net


class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8):
        super(DeepGuidedFilter, self).__init__()
        self.lr = build_lr_net()
        #self.gf = FastGuidedFilter(radius, eps)
        #self.gf = GuidedFilter(radius, eps)
        self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm)

    def forward(self, x_lr, x_hr):
        residual_lr = self.lr(x_lr)
        residual_hr = self.gf(x_lr, residual_lr, x_hr)
        return (x_hr + residual_hr).clamp(0, 1)
        #return self.gf(x_hr, self.lr(x_hr)).clamp(0, 1)

    def init_lr(self, path):
        self.lr.load_state_dict(torch.load(path))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

downsample_folder_test = '/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/down_sample_2_test_log'
input_folder_test = '/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/input_test_log'

#label_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRv1_data/testlabel_log'
label_test = '/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/testlabel_log'

# downsample_folder_test = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRV1/down_sample_2_train_log" ## yes
# input_folder_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRV1/input_train_log'

# label_test = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRv1_data/trainlabel_log"

# 模型文件
#model_path = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRV1/models/convguided_filter_best_hdrv1_r1_lanczos_log(2).pth"
#model_path = "/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/models/convguided_filter_best_hdrv1_r1_lanczos_log(2).pth"
#model_path = "/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/models_retrain/convguided_filter_best_hdr_r1_lanczos_log.pth"
#model_path = "/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/models_retrain/convguided_filter_best_lavel.pth"
model_path = "/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/models_retrain/convguided_filter_residual_best_mix_residual.pth"

# 输出文件夹
#output_dir = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRV1/model_testoutput/origin_log_v2"
output_dir = "/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/result/df_output_s2r_residual"
#output_dir_ls = "/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/down_sample_2_test_log_filtered_origin"

os.makedirs(output_dir, exist_ok=True)

# 加载模型
model = DeepGuidedFilter(radius=1, eps=1e-8) ##### 根据训练 r 的不同进行调整
model = nn.DataParallel(model).to(device)  # 先包 DataParallel

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)          # 再加载参数

model.eval()

class GuidedFilterDataset(Dataset):
    def __init__(self, x_lr_dir, x_hr_dir, label_dir):
        self.x_lr_files = sorted([os.path.join(x_lr_dir, f) for f in os.listdir(x_lr_dir)])
        self.x_hr_files = sorted([os.path.join(x_hr_dir, f) for f in os.listdir(x_hr_dir)])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        print(f"LR: {len(self.x_lr_files)}, HR: {len(self.x_hr_files)}, Label: {len(self.label_files)}")

    def __len__(self):
        return min(len(self.x_lr_files), len(self.x_hr_files), len(self.label_files))

    def __getitem__(self, idx):
        x_lr_path = self.x_lr_files[idx]
        x_hr_path = self.x_hr_files[idx]
        label_path = self.label_files[idx]

        x_lr   = cv2.imread(self.x_lr_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        x_hr   = cv2.imread(self.x_hr_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        label  = cv2.imread(self.label_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        input_hr_h, input_hr_w = x_hr.shape
        input_lr_h, input_lr_w = x_lr.shape

        filename = os.path.splitext(os.path.basename(x_hr_path))[0]  # 用 HR 图的文件名作为基准

        return {
            'input_lr': torch.from_numpy(x_lr).unsqueeze(0).unsqueeze(0),
            'input_hr': torch.from_numpy(x_hr).unsqueeze(0).unsqueeze(0),
            'target_img': torch.from_numpy(label).unsqueeze(0).unsqueeze(0),
            'filename': filename,
            'input_height_hr': input_hr_h,
            'input_width_hr': input_hr_w,
            'input_height_lr': input_lr_h,
            'input_width_lr': input_lr_w
        }

def var_custom_collate(batch):
    min_h_hr = min([item['input_height_hr'] for item in batch])
    min_w_hr = min([item['input_width_hr'] for item in batch])
    min_h_lr = min([item['input_height_lr'] for item in batch])
    min_w_lr = min([item['input_width_lr'] for item in batch])

    batch_x_hr = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_x_lr = torch.zeros(len(batch), 1, min_h_lr, min_w_lr)
    batch_target_images = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)

    filenames = []

    for idx, item in enumerate(batch):
        x_lr_crop = item['input_lr'][:, :, :min_h_lr, :min_w_lr]
        x_hr_crop = item['input_hr'][:, :, :min_h_hr, :min_w_hr]
        label_crop = item['target_img'][:, :, :min_h_hr, :min_w_hr]

        batch_x_lr[idx] = x_lr_crop
        batch_x_hr[idx] = x_hr_crop
        batch_target_images[idx] = label_crop
        filenames.append(item['filename'])

    return batch_x_lr, batch_x_hr, batch_target_images, filenames

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
    
test_dataset = GuidedFilterDataset(
    x_lr_dir = downsample_folder_test,
    x_hr_dir = input_folder_test,
    label_dir = label_test
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0,
                          collate_fn=var_custom_collate, pin_memory=True, drop_last=False)

criterion = torch.nn.MSELoss(reduction='mean')

def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1.0 / mse)
    
    return psnr, mse

def normalize(img):
    min_val = img.min()
    max_val = img.max()
    return (img - min_val) / (max_val - min_val + 1e-8)
def norm_and_save(tensor, path):
    # 转为 CPU numpy 数组
    t = tensor.detach().cpu().numpy()
    # 去掉 batch/channel 维度 (支持单通道图像)
    if t.ndim == 4:  # [B,C,H,W]
        t = t[0]
    if t.ndim == 3 and t.shape[0] == 1:  # [1,H,W]
        t = t[0]

    # 确保在 [0,1] 范围内
    t = np.clip(t, 0, 1)

    # 转为 [0,255] uint8
    t = (t * 255.0).round().astype(np.uint8)

    # 保存为 PNG
    cv2.imwrite(path, t)

# def save_float32_tiff(tensor, path):
#     """
#     将 tensor 保存为 float32 精度的 tiff 文件。
#     - 不做 clip/归一化/量化
#     - 保留多通道（1通道 或 3通道）
#     """
#     # 转为 numpy，去 batch/channel 维度
#     t = tensor.detach().cpu().numpy()
#     if t.ndim == 4:       # [B,C,H,W]
#         t = t[0]
#     if t.ndim == 3 and t.shape[0] == 1:  # [1,H,W]
#         t = t[0]
#     if t.ndim == 3 and t.shape[0] == 3:  # [3,H,W] → [H,W,3]
#         t = np.transpose(t, (1, 2, 0))   # [H,W,C]

#     # 检查浮点类型
#     if t.dtype != np.float32:
#         t = t.astype(np.float32)

#     # 确保路径合法
#     os.makedirs(os.path.dirname(path), exist_ok=True)

#     # 保存为 float32 tiff
#     cv2.imwrite(path, t)   # 自动使用 .tiff 保存为 float32

def save_float32_tiff(img, save_path):
    """
    直接保存原始 img 为 float32 tiff，不做归一化。
    保留负值和原始动态范围。
    """
    img = img.detach().cpu().numpy()
    # 转为 numpy，去 batch/channel 维度
    if img.ndim == 4:       # [B,C,H,W]
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 1:  # [1,H,W]
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 3:  # [3,H,W] → [H,W,3]
        img = np.transpose(img, (1, 2, 0))
           # [H,W,C]
    img = img.astype(np.float32)
    imageio.imwrite(save_path, img)

# 开始测试
import time
test_psnr, test_mse = 0, 0
total_time = 0
total_images = 0

with torch.no_grad():
    for test_x_lr, test_x_hr, test_label, filenames in tqdm(test_loader, desc="Testing"):
        #test_x_lr = crop_to_even(test_x_lr)
        test_x_hr = crop_to_even(test_x_hr)
        test_label = crop_to_even(test_label)

        test_x_lr, test_x_hr, test_label = test_x_lr.to(device), test_x_hr.to(device), test_label.to(device)
        # # ========== Normalize ==========
        # test_x_lr = tonemap_mu(test_x_lr, mu=1000.0)
        # test_x_hr = tonemap_mu(test_x_hr, mu=1000.0)
        # =======================================
        test_x_lr = test_x_lr - 0.5
        test_x_hr = test_x_hr - 0.5


        start_time = time.time()
        
        test_output = model(test_x_lr, test_x_hr)
        #lr_output = model.module.lr(test_x_lr)
        #test_output = inverse_tonemap_mu(test_output, mu=1000.0)
        #test_output = model(test_x_lr, test_x_hr)
        #test_output = model(test_x_hr)
        torch.cuda.synchronize()  # 若使用GPU，确保所有CUDA操作完成
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        total_images += test_output.size(0)

        v_loss = criterion(test_output, test_label)

        # test_output = normalize(test_output)
        # test_label = normalize(test_label)
        psnr, mse = calc_psnr(test_output, test_label)
        test_mse += mse
        test_psnr += psnr

        #test_output = test_output /255
        for i, fname in enumerate(filenames):
            #norm_and_save(test_output[i], f"{output_dir}/{fname}_pred.png")
            save_float32_tiff(test_output[i], f"{output_dir}/{fname}_pred.tiff" )

            #save_float32_tiff(lr_output[i].detach().cpu(),f"{output_dir_ls}/{fname}_lr.tiff")

    avg_test_mse = test_mse / len(test_loader)
    avg_test_psnr = test_psnr / len(test_loader)
    avg_time_per_image = total_time / total_images

print(f"Test Finished - Average PSNR: {avg_test_psnr:.2f} dB, Average MSE: {avg_test_mse:.6f}")
print(f"Average Inference Time per Image: {avg_time_per_image * 1000:.2f} ms")