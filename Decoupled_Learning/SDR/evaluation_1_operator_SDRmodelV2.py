import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.data as data
import math

from torch.backends import cudnn
from torch.autograd import Variable

import cv2
import skimage
from skimage import io
from os import listdir
from os.path import join
from PIL import Image
import os
import time
import glob, os
import tifffile
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def save_mse_mask(pred_numpy, label_numpy, save_path):
    """
    可视化并保存 MSE 热力图，误差越大颜色越红。
    输入图像应为 (H, W) 或 (1, H, W) 格式的 numpy 数组。
    """
    # 去除 batch 维
    if pred_numpy.ndim == 3 and pred_numpy.shape[0] == 1:
        pred_numpy = pred_numpy[0]
    if label_numpy.ndim == 3 and label_numpy.shape[0] == 1:
        label_numpy = label_numpy[0]

    # 计算像素级误差
    error_map = pred_numpy - label_numpy
    error_map = np.abs(error_map)

    # 归一化到 0-1，用于 colormap 显示
    error_norm = error_map / np.max(error_map + 1e-8)

    # 可视化
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(error_map, cmap='hot')  # 或者 'inferno', 'jet', 'magma'
    plt.colorbar(label='Normalized MSE')
    plt.tight_layout()

    # 保存为 png
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"[保存成功] 误差热力图已保存至: {save_path}")

# def save_image(filename, data):
#     #img = data.cpu().numpy().astype(np.float32)  # 保证 float32 精度
#     img = np.ascontiguousarray(data)
#     if img.ndim == 3 and img.shape[0] == 1:
#         img = img[0]  # (1, H, W) → (H, W)

#     assert img.ndim == 2, "图像必须为单通道 (H, W) 格式"

#     tifffile.imwrite(
#         filename,
#         img,
#         photometric='minisblack',   # 灰度图
#         dtype=np.float32,
#         planarconfig='contig',
#         compression=None
#     )

def save_image(filename, data):
    img = data.numpy()
    img = np.clip(img,0,255)
    img = img.transpose(1, 2, 0).astype("uint8")

    cv2.imwrite(filename,img)

def reflect_conv2d(x, weight, bias=None, stride=1, padding=1, dilation=1):
    pad_total = padding
    x = F.pad(x, (pad_total, pad_total, pad_total, pad_total), mode='reflect')
    return F.conv2d(x, weight, bias=bias, stride=stride, padding=0, dilation=dilation)

def zero_conv2d(input, weight, bias=None, stride=1, padding=None, dilation=1, groups=1):
    """
    支持动态卷积核大小的zero padding卷积
    - input: (N, C_in, H, W)
    - weight: (C_out, C_in, kH, kW)
    - padding: 若None则自动根据kernel size设置合适的padding
    """
    kH, kW = weight.shape[-2:]
    if padding is None:
        # 自动选择合适padding
        pad_h = (kH - 1) // 2
        pad_w = (kW - 1) // 2
        padding = (pad_h, pad_w)
    if isinstance(padding, int):
        padding = (padding, padding)
    out = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    return out

def visualize_and_save(feat, layer_name, batch_idx):
    vis_feat = feat[0, 0].detach().cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.imshow(vis_feat, cmap='gray')
    plt.title(f'{layer_name}_batch{batch_idx}')
    #plt.colorbar()
    
    save_dir = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/pre_exp"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f"{layer_name}_visual.png"))
    plt.close()

def center_crop(tensor, target):
    # 将tensor裁剪成和target一样的大小（中心对齐）
    _, _, h, w = tensor.shape
    _, _, th, tw = target.shape
    dh = (h - th) // 2
    dw = (w - tw) // 2
    return tensor[..., dh:dh+th, dw:dw+tw]

class Resize(nn.Module):
    def __init__(self, a,b,c,d):
        super(Resize, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def forward(self, x):
        x = x.resize(self.a,self.b,self.c,self.d)
        return x

class EdgeComputation(nn.Module):
    def __init__(self):
        super(EdgeComputation, self).__init__()

    def forward(self, x):
        x_diffx = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])
        x_diffy = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])

        y = torch.Tensor(x.size()).cuda()
        y.fill_(0)
        y[:,:,:,1:] += x_diffx
        y[:,:,:,:-1] += x_diffx
        y[:,:,1:,:] += x_diffy
        y[:,:,:-1,:] += x_diffy
        y = torch.sum(y,1)/3
        y /= 4
        return y
'''
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.relu = nn.ReLU()
        self.resize1 = Resize(64,2,3,3)
        self.resize2 = Resize(64,64,3,3)
        self.resize3 = Resize(1,64,3,3)
        self.resize4 = Resize(64,64,4,4)

        self.fc1 = nn.Linear(1,1152)
        self.fc2 = nn.Linear(1,36864)
        self.fc3 = nn.Linear(1,36864)
        self.fc4 = nn.Linear(1,36864)
        self.fc5 = nn.Linear(1,36864)
        self.fc6 = nn.Linear(1,36864)
        self.fc7 = nn.Linear(1,36864)
        self.fc8 = nn.Linear(1,36864)
        self.fc9 = nn.Linear(1,36864)
        self.fc10 = nn.Linear(1,36864)
        self.fc11 = nn.Linear(1,36864)
        self.fc12 = nn.Linear(1,36864)
        self.fc13 = nn.Linear(1,36864)
        self.fc14 = nn.Linear(1,36864)
        self.fc15 = nn.Linear(1,36864)
        self.fc16 = nn.Linear(1,36864)
        self.fc17 = nn.Linear(1,36864)
        self.fc18 = nn.Linear(1,65536)
        self.fc19 = nn.Linear(1,36864)
        self.fc20 = nn.Linear(1,576)
        
        self.norm_1 = nn.InstanceNorm2d(64, affine=True)
        self.norm_2 = nn.InstanceNorm2d(64, affine=True)
        self.norm_3 = nn.InstanceNorm2d(64, affine=True)
        self.norm_4 = nn.InstanceNorm2d(64, affine=True)
        self.norm_5 = nn.InstanceNorm2d(64, affine=True)
        self.norm_6 = nn.InstanceNorm2d(64, affine=True)
        self.norm_7 = nn.InstanceNorm2d(64, affine=True)
        self.norm_8 = nn.InstanceNorm2d(64, affine=True)
        self.norm_9 = nn.InstanceNorm2d(64, affine=True)
        self.norm_10 = nn.InstanceNorm2d(64, affine=True)
        self.norm_11 = nn.InstanceNorm2d(64, affine=True)
        self.norm_12 = nn.InstanceNorm2d(64, affine=True)
        self.norm_13 = nn.InstanceNorm2d(64, affine=True)
        self.norm_14 = nn.InstanceNorm2d(64, affine=True)
        self.norm_15 = nn.InstanceNorm2d(64, affine=True)
        self.norm_16 = nn.InstanceNorm2d(64, affine=True)
        self.norm_17 = nn.InstanceNorm2d(64, affine=True)
        self.norm_18 = nn.InstanceNorm2d(64, affine=True)
        self.norm_19 = nn.InstanceNorm2d(64, affine=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.ConvTranspose3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        x_param = input[0]
        x_img = input[1]

        conv1_weight = self.fc1(x_param)
        conv2_weight = self.fc2(x_param)
        conv3_weight = self.fc3(x_param)
        conv4_weight = self.fc4(x_param)
        conv5_weight = self.fc5(x_param)
        conv6_weight = self.fc6(x_param)
        conv7_weight = self.fc7(x_param)
        conv8_weight = self.fc8(x_param)
        conv9_weight = self.fc9(x_param)
        conv10_weight = self.fc10(x_param)
        conv11_weight = self.fc11(x_param)
        conv12_weight = self.fc12(x_param)
        conv13_weight = self.fc13(x_param)
        conv14_weight = self.fc14(x_param)
        conv15_weight = self.fc15(x_param)
        conv16_weight = self.fc16(x_param)
        conv17_weight = self.fc17(x_param)
        conv18_weight = self.fc18(x_param)
        conv19_weight = self.fc19(x_param)
        conv20_weight = self.fc20(x_param)

        batch_size = x_img.size(0)
        batch_outputs = []
        for n in range(batch_size):
            conv1_feat = self.relu(self.norm_1(
                reflect_conv2d(x_img.narrow(0, n, 1), self.resize1(conv1_weight[n]), bias=None, padding=1)))
            visualize_and_save(conv1_feat, 'conv1', n)
            conv2_feat = self.relu(self.norm_2(
                reflect_conv2d(conv1_feat, self.resize2(conv2_weight[n]), bias=None, padding=1)))
            visualize_and_save(conv2_feat, 'conv2', n)
            conv3_feat = self.relu(self.norm_3(
                reflect_conv2d(conv2_feat, self.resize2(conv3_weight[n]), bias=None, stride=2, padding=1)))
            visualize_and_save(conv3_feat, 'conv3', n)
            # residual block
            conv4_feat = self.relu(self.norm_4(
                zero_conv2d(conv3_feat, self.resize2(conv4_weight[n]), bias=None, stride=1, padding=2, dilation=2)))
            visualize_and_save(conv4_feat, 'conv4', n)
            conv5_feat = self.norm_5(
                zero_conv2d(conv4_feat, self.resize2(conv5_weight[n]), bias=None, stride=1, padding=2, dilation=2))
            visualize_and_save(conv5_feat, 'conv5_pre_res', n)
            # ---- 残差加法前center crop ----
            if conv3_feat.shape != conv5_feat.shape:
                if conv3_feat.shape[2:] != conv5_feat.shape[2:]:
                    if conv3_feat.shape[2] > conv5_feat.shape[2] or conv3_feat.shape[3] > conv5_feat.shape[3]:
                        conv3_feat_crop = center_crop(conv3_feat, conv5_feat)
                        conv5_feat = self.relu(conv3_feat_crop + conv5_feat)
                    else:
                        conv5_feat_crop = center_crop(conv5_feat, conv3_feat)
                        conv5_feat = self.relu(conv3_feat + conv5_feat_crop)
                        visualize_and_save(conv5_feat, 'conv5_post_res', n)
                else:
                    conv5_feat = self.relu(conv3_feat + conv5_feat)
                    visualize_and_save(conv5_feat, 'conv5_post_res', n)
            else:
                conv5_feat = self.relu(conv3_feat + conv5_feat)
                visualize_and_save(conv5_feat, 'conv5_post_res', n)

            # residual block
            conv6_feat = self.relu(self.norm_6(
                zero_conv2d(conv5_feat, self.resize2(conv6_weight[n]), bias=None, stride=1, padding=4, dilation=4)))
            visualize_and_save(conv6_feat, 'conv6', n)
            conv7_feat = self.norm_7(
                zero_conv2d(conv6_feat, self.resize2(conv7_weight[n]), bias=None, stride=1, padding=4, dilation=4))
            visualize_and_save(conv7_feat, 'conv7_pre_res', n)
            if conv5_feat.shape != conv7_feat.shape:
                if conv5_feat.shape[2] > conv7_feat.shape[2] or conv5_feat.shape[3] > conv7_feat.shape[3]:
                    conv5_feat_crop = center_crop(conv5_feat, conv7_feat)
                    conv7_feat = self.relu(conv5_feat_crop + conv7_feat)
                    visualize_and_save(conv7_feat, 'conv7_post_res', n)
                else:
                    conv7_feat_crop = center_crop(conv7_feat, conv5_feat)
                    conv7_feat = self.relu(conv5_feat + conv7_feat_crop)
                    visualize_and_save(conv7_feat, 'conv7_post_res', n)
            else:
                conv7_feat = self.relu(conv5_feat + conv7_feat)
                visualize_and_save(conv7_feat, 'conv7_post_res', n)

            # residual block
            conv8_feat = self.relu(self.norm_8(
                zero_conv2d(conv7_feat, self.resize2(conv8_weight[n]), bias=None, stride=1, padding=4, dilation=4)))
            visualize_and_save(conv8_feat, 'conv8', n)
            conv9_feat = self.norm_9(
                zero_conv2d(conv8_feat, self.resize2(conv9_weight[n]), bias=None, stride=1, padding=4, dilation=4))
            visualize_and_save(conv9_feat, 'conv9', n)
            if conv7_feat.shape != conv9_feat.shape:
                if conv7_feat.shape[2] > conv9_feat.shape[2] or conv7_feat.shape[3] > conv9_feat.shape[3]:
                    conv7_feat_crop = center_crop(conv7_feat, conv9_feat)
                    conv9_feat = self.relu(conv7_feat_crop + conv9_feat)
                    visualize_and_save(conv9_feat, 'conv9_post_res', n)
                else:
                    conv9_feat_crop = center_crop(conv9_feat, conv7_feat)
                    conv9_feat = self.relu(conv7_feat + conv9_feat_crop)
                    visualize_and_save(conv9_feat, 'conv9_post_res', n)
            else:
                conv9_feat = self.relu(conv7_feat + conv9_feat)
                visualize_and_save(conv9_feat, 'conv9_post_res', n)

            # residual block
            conv10_feat = self.relu(self.norm_10(
                zero_conv2d(conv9_feat, self.resize2(conv10_weight[n]), bias=None, stride=1, padding=8, dilation=8)))
            visualize_and_save(conv10_feat, 'conv10', n)
            conv11_feat = self.norm_11(
                zero_conv2d(conv10_feat, self.resize2(conv11_weight[n]), bias=None, stride=1, padding=8, dilation=8))
            visualize_and_save(conv11_feat, 'conv11', n)
            if conv9_feat.shape != conv11_feat.shape:
                if conv9_feat.shape[2] > conv11_feat.shape[2] or conv9_feat.shape[3] > conv11_feat.shape[3]:
                    conv9_feat_crop = center_crop(conv9_feat, conv11_feat)
                    conv11_feat = self.relu(conv9_feat_crop + conv11_feat)
                    visualize_and_save(conv11_feat, 'conv11_post_res', n)
                else:
                    conv11_feat_crop = center_crop(conv11_feat, conv9_feat)
                    conv11_feat = self.relu(conv9_feat + conv11_feat_crop)
                    visualize_and_save(conv11_feat, 'conv11_post_res', n)
            else:
                conv11_feat = self.relu(conv9_feat + conv11_feat)
                visualize_and_save(conv11_feat, 'conv11_post_res', n)

            # residual block
            conv12_feat = self.relu(self.norm_12(
                zero_conv2d(conv11_feat, self.resize2(conv12_weight[n]), bias=None, stride=1, padding=8, dilation=8)))
            visualize_and_save(conv12_feat, 'conv12', n)
            conv13_feat = self.norm_13(
                zero_conv2d(conv12_feat, self.resize2(conv13_weight[n]), bias=None, stride=1, padding=8, dilation=8))
            visualize_and_save(conv13_feat, 'conv13', n)
            if conv11_feat.shape != conv13_feat.shape:
                if conv11_feat.shape[2] > conv13_feat.shape[2] or conv11_feat.shape[3] > conv13_feat.shape[3]:
                    conv11_feat_crop = center_crop(conv11_feat, conv13_feat)
                    conv13_feat = self.relu(conv11_feat_crop + conv13_feat)
                    visualize_and_save(conv13_feat, 'conv13_post_res', n)
                else:
                    conv13_feat_crop = center_crop(conv13_feat, conv11_feat)
                    conv13_feat = self.relu(conv11_feat + conv13_feat_crop)
                    visualize_and_save(conv13_feat, 'conv13_post_res', n)
            else:
                conv13_feat = self.relu(conv11_feat + conv13_feat)
                visualize_and_save(conv13_feat, 'conv13_post_res', n)

            # residual block
            conv14_feat = self.relu(self.norm_14(
                zero_conv2d(conv13_feat, self.resize2(conv14_weight[n]), bias=None, stride=1, padding=16, dilation=16)))
            visualize_and_save(conv14_feat, 'conv14', n)
            conv15_feat = self.norm_15(
                zero_conv2d(conv14_feat, self.resize2(conv15_weight[n]), bias=None, stride=1, padding=16, dilation=16))
            visualize_and_save(conv15_feat, 'conv15', n)
            if conv13_feat.shape != conv15_feat.shape:
                if conv13_feat.shape[2] > conv15_feat.shape[2] or conv13_feat.shape[3] > conv15_feat.shape[3]:
                    conv13_feat_crop = center_crop(conv13_feat, conv15_feat)
                    conv15_feat = self.relu(conv13_feat_crop + conv15_feat)
                    visualize_and_save(conv15_feat, 'conv15_post_res', n)
                else:
                    conv15_feat_crop = center_crop(conv15_feat, conv13_feat)
                    conv15_feat = self.relu(conv13_feat + conv15_feat_crop)
                    visualize_and_save(conv15_feat, 'conv15_post_res', n)
            else:
                conv15_feat = self.relu(conv13_feat + conv15_feat)
                visualize_and_save(conv15_feat, 'conv15_post_res', n)

            # residual block
            conv16_feat = self.relu(self.norm_16(
                zero_conv2d(conv15_feat, self.resize2(conv16_weight[n]), bias=None, stride=1, padding=1, dilation=1)))
            visualize_and_save(conv16_feat, 'conv16', n)
            conv17_feat = self.norm_17(
                zero_conv2d(conv16_feat, self.resize2(conv17_weight[n]), bias=None, stride=1, padding=1, dilation=1))
            visualize_and_save(conv17_feat, 'conv17', n)
            if conv15_feat.shape != conv17_feat.shape:
                if conv15_feat.shape[2] > conv17_feat.shape[2] or conv15_feat.shape[3] > conv17_feat.shape[3]:
                    conv15_feat_crop = center_crop(conv15_feat, conv17_feat)
                    conv17_feat = self.relu(conv15_feat_crop + conv17_feat)
                    visualize_and_save(conv17_feat, 'conv17_post_res', n)
                else:
                    conv17_feat_crop = center_crop(conv17_feat, conv15_feat)
                    conv17_feat = self.relu(conv15_feat + conv17_feat_crop)
                    visualize_and_save(conv17_feat, 'conv17_post_res', n)
            else:
                conv17_feat = self.relu(conv15_feat + conv17_feat)
                visualize_and_save(conv17_feat, 'conv17_post_res', n)

            # last 3 convs
            upsampled = F.interpolate(conv17_feat, scale_factor=2, mode='nearest')
            upsampled_padded = F.pad(upsampled, pad=(1,1,1,1), mode='reflect')
            conv18_feat = self.relu(
                self.norm_18(
                    F.conv2d(
                        upsampled_padded,
                        self.resize4(conv18_weight[n]),
                        bias=None,
                        stride=1,
                        padding=0
                    )
                )
            )
            visualize_and_save(conv18_feat, 'conv18', n)
            conv19_feat = self.relu(self.norm_19(
                reflect_conv2d(conv18_feat, self.resize2(conv19_weight[n]), bias=None, stride=1, padding=1)))
            visualize_and_save(conv19_feat, 'conv19', n)
            output = self.relu(
                reflect_conv2d(conv19_feat, self.resize3(conv20_weight[n]), bias=None, stride=1, padding=1))

            batch_outputs.append(output)

        output = torch.cat(batch_outputs, dim=0)
        return output
'''

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.relu = nn.ReLU()
        self.resize1 = Resize(64,2,3,3)
        self.resize2 = Resize(64,64,3,3)
        self.resize3 = Resize(1,64,3,3)
        self.resize4 = Resize(64,64,4,4)

        self.fc1 = nn.Linear(1,1152)
        self.fc2 = nn.Linear(1,36864)
        self.fc3 = nn.Linear(1,36864)
        # conv4,5：3x3
        self.fc4 = nn.Linear(1, 36864)
        self.fc5 = nn.Linear(1, 36864)
        self.resize4 = Resize(64, 64, 3, 3)
        self.resize5 = Resize(64, 64, 3, 3)

        # conv6,7：5x5
        self.fc6 = nn.Linear(1, 102400)
        self.fc7 = nn.Linear(1, 102400)
        self.resize6 = Resize(64, 64, 5, 5)
        self.resize7 = Resize(64, 64, 5, 5)

        # conv8,9：7x7
        self.fc8 = nn.Linear(1, 200704)
        self.fc9 = nn.Linear(1, 200704)
        self.resize8 = Resize(64, 64, 7, 7)
        self.resize9 = Resize(64, 64, 7, 7)

        # conv10,11：9x9
        self.fc10 = nn.Linear(1, 331776)
        self.fc11 = nn.Linear(1, 331776)
        self.resize10 = Resize(64, 64, 9, 9)
        self.resize11 = Resize(64, 64, 9, 9)

        # conv12,13：7x7
        self.fc12 = nn.Linear(1, 331776)
        self.fc13 = nn.Linear(1, 331776)
        self.resize12 = Resize(64, 64, 9, 9)
        self.resize13 = Resize(64, 64, 9, 9)

        # conv14,15：5x5
        self.fc14 = nn.Linear(1, 921600)
        self.fc15 = nn.Linear(1, 921600)
        self.resize14 = Resize(64, 64, 15, 15)
        self.resize15 = Resize(64, 64, 15, 15)

        # conv16,17：3x3
        self.fc16 = nn.Linear(1, 36864)
        self.fc17 = nn.Linear(1, 36864)
        self.resize16 = Resize(64, 64, 3, 3)
        self.resize17 = Resize(64, 64, 3, 3)
        self.fc18 = nn.Linear(1,65536)
        self.resize18 = Resize(64, 64, 4, 4)
        self.fc19 = nn.Linear(1,36864)
        self.fc20 = nn.Linear(1,576)
        
        self.norm_1 = nn.InstanceNorm2d(64, affine=True)
        self.norm_2 = nn.InstanceNorm2d(64, affine=True)
        self.norm_3 = nn.InstanceNorm2d(64, affine=True)
        self.norm_4 = nn.InstanceNorm2d(64, affine=True)
        self.norm_5 = nn.InstanceNorm2d(64, affine=True)
        self.norm_6 = nn.InstanceNorm2d(64, affine=True)
        self.norm_7 = nn.InstanceNorm2d(64, affine=True)
        self.norm_8 = nn.InstanceNorm2d(64, affine=True)
        self.norm_9 = nn.InstanceNorm2d(64, affine=True)
        self.norm_10 = nn.InstanceNorm2d(64, affine=True)
        self.norm_11 = nn.InstanceNorm2d(64, affine=True)
        self.norm_12 = nn.InstanceNorm2d(64, affine=True)
        self.norm_13 = nn.InstanceNorm2d(64, affine=True)
        self.norm_14 = nn.InstanceNorm2d(64, affine=True)
        self.norm_15 = nn.InstanceNorm2d(64, affine=True)
        self.norm_16 = nn.InstanceNorm2d(64, affine=True)
        self.norm_17 = nn.InstanceNorm2d(64, affine=True)
        self.norm_18 = nn.InstanceNorm2d(64, affine=True)
        self.norm_19 = nn.InstanceNorm2d(64, affine=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.ConvTranspose3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        x_param = input[0]
        x_img = input[1]

        conv1_weight = self.fc1(x_param)
        conv2_weight = self.fc2(x_param)
        conv3_weight = self.fc3(x_param)
        conv4_weight = self.fc4(x_param)
        conv5_weight = self.fc5(x_param)
        conv6_weight = self.fc6(x_param)
        conv7_weight = self.fc7(x_param)
        conv8_weight = self.fc8(x_param)
        conv9_weight = self.fc9(x_param)
        conv10_weight = self.fc10(x_param)
        conv11_weight = self.fc11(x_param)
        conv12_weight = self.fc12(x_param)
        conv13_weight = self.fc13(x_param)
        conv14_weight = self.fc14(x_param)
        conv15_weight = self.fc15(x_param)
        conv16_weight = self.fc16(x_param)
        conv17_weight = self.fc17(x_param)
        conv18_weight = self.fc18(x_param)
        conv19_weight = self.fc19(x_param)
        conv20_weight = self.fc20(x_param)

        batch_size = x_img.size(0)
        batch_outputs = []
        for n in range(batch_size):
            conv1_feat = self.relu(self.norm_1(
                reflect_conv2d(x_img.narrow(0, n, 1), self.resize1(conv1_weight[n]), bias=None, padding=1)))
            visualize_and_save(conv1_feat, 'conv1', n)
            conv2_feat = self.relu(self.norm_2(
                reflect_conv2d(conv1_feat, self.resize2(conv2_weight[n]), bias=None, padding=1)))
            visualize_and_save(conv2_feat, 'conv2', n)
            conv3_feat = self.relu(self.norm_3(
                reflect_conv2d(conv2_feat, self.resize2(conv3_weight[n]), bias=None, stride=2, padding=1)))
            visualize_and_save(conv3_feat, 'conv3', n)
            # residual block
            conv4_feat = self.relu(self.norm_4(
                zero_conv2d(conv3_feat, self.resize4(conv4_weight[n]), bias=None, stride=1, padding=1)))  # 或padding=None自动推断
            visualize_and_save(conv4_feat, 'conv4', n)
            conv5_feat = self.norm_5(
                zero_conv2d(conv4_feat, self.resize5(conv5_weight[n]), bias=None, stride=1, padding=1))
            visualize_and_save(conv5_feat, 'conv5_pre_res', n)
            # ---- 残差加法前center crop ----
            if conv3_feat.shape != conv5_feat.shape:
                if conv3_feat.shape[2:] != conv5_feat.shape[2:]:
                    if conv3_feat.shape[2] > conv5_feat.shape[2] or conv3_feat.shape[3] > conv5_feat.shape[3]:
                        conv3_feat_crop = center_crop(conv3_feat, conv5_feat)
                        conv5_feat = self.relu(conv3_feat_crop + conv5_feat)
                        visualize_and_save(conv5_feat, 'conv5_post_res', n)
                    else:
                        conv5_feat_crop = center_crop(conv5_feat, conv3_feat)
                        conv5_feat = self.relu(conv3_feat + conv5_feat_crop)
                        visualize_and_save(conv5_feat, 'conv5_post_res', n)
                else:
                    conv5_feat = self.relu(conv3_feat + conv5_feat)
                    visualize_and_save(conv5_feat, 'conv5_post_res', n)
            else:
                conv5_feat = self.relu(conv3_feat + conv5_feat)
                visualize_and_save(conv5_feat, 'conv5_post_res', n)

            # residual block
            conv6_feat = self.relu(self.norm_6(
                zero_conv2d(conv5_feat, self.resize6(conv6_weight[n]), bias=None, stride=1, padding=2)))
            conv7_feat = self.norm_7(
                zero_conv2d(conv6_feat, self.resize7(conv7_weight[n]), bias=None, stride=1, padding=2))
            if conv5_feat.shape != conv7_feat.shape:
                if conv5_feat.shape[2] > conv7_feat.shape[2] or conv5_feat.shape[3] > conv7_feat.shape[3]:
                    conv5_feat_crop = center_crop(conv5_feat, conv7_feat)
                    conv7_feat = self.relu(conv5_feat_crop + conv7_feat)
                else:
                    conv7_feat_crop = center_crop(conv7_feat, conv5_feat)
                    conv7_feat = self.relu(conv5_feat + conv7_feat_crop)
            else:
                conv7_feat = self.relu(conv5_feat + conv7_feat)

            # residual block
            conv8_feat = self.relu(self.norm_8(
                zero_conv2d(conv7_feat, self.resize8(conv8_weight[n]), bias=None, stride=1, padding=3)))
            conv9_feat = self.norm_9(
                zero_conv2d(conv8_feat, self.resize9(conv9_weight[n]), bias=None, stride=1, padding=3))
            if conv7_feat.shape != conv9_feat.shape:
                if conv7_feat.shape[2] > conv9_feat.shape[2] or conv7_feat.shape[3] > conv9_feat.shape[3]:
                    conv7_feat_crop = center_crop(conv7_feat, conv9_feat)
                    conv9_feat = self.relu(conv7_feat_crop + conv9_feat)
                else:
                    conv9_feat_crop = center_crop(conv9_feat, conv7_feat)
                    conv9_feat = self.relu(conv7_feat + conv9_feat_crop)
            else:
                conv9_feat = self.relu(conv7_feat + conv9_feat)

            # residual block
            conv10_feat = self.relu(self.norm_10(
                zero_conv2d(conv9_feat, self.resize10(conv10_weight[n]), bias=None, stride=1, padding=4)))
            conv11_feat = self.norm_11(
                zero_conv2d(conv10_feat, self.resize11(conv11_weight[n]), bias=None, stride=1, padding=4))
            if conv9_feat.shape != conv11_feat.shape:
                if conv9_feat.shape[2] > conv11_feat.shape[2] or conv9_feat.shape[3] > conv11_feat.shape[3]:
                    conv9_feat_crop = center_crop(conv9_feat, conv11_feat)
                    conv11_feat = self.relu(conv9_feat_crop + conv11_feat)
                else:
                    conv11_feat_crop = center_crop(conv11_feat, conv9_feat)
                    conv11_feat = self.relu(conv9_feat + conv11_feat_crop)
            else:
                conv11_feat = self.relu(conv9_feat + conv11_feat)

            # residual block
            conv12_feat = self.relu(self.norm_12(
                zero_conv2d(conv11_feat, self.resize12(conv12_weight[n]), bias=None, stride=1, padding=4)))
            conv13_feat = self.norm_13(
                zero_conv2d(conv12_feat, self.resize13(conv13_weight[n]), bias=None, stride=1, padding=4))
            if conv11_feat.shape != conv13_feat.shape:
                if conv11_feat.shape[2] > conv13_feat.shape[2] or conv11_feat.shape[3] > conv13_feat.shape[3]:
                    conv11_feat_crop = center_crop(conv11_feat, conv13_feat)
                    conv13_feat = self.relu(conv11_feat_crop + conv13_feat)
                else:
                    conv13_feat_crop = center_crop(conv13_feat, conv11_feat)
                    conv13_feat = self.relu(conv11_feat + conv13_feat_crop)
            else:
                conv13_feat = self.relu(conv11_feat + conv13_feat)

            # residual block
            conv14_feat = self.relu(self.norm_14(
                zero_conv2d(conv13_feat, self.resize14(conv14_weight[n]), bias=None, stride=1, padding=7)))
            conv15_feat = self.norm_15(
                zero_conv2d(conv14_feat, self.resize15(conv15_weight[n]), bias=None, stride=1, padding=7))
            if conv13_feat.shape != conv15_feat.shape:
                if conv13_feat.shape[2] > conv15_feat.shape[2] or conv13_feat.shape[3] > conv15_feat.shape[3]:
                    conv13_feat_crop = center_crop(conv13_feat, conv15_feat)
                    conv15_feat = self.relu(conv13_feat_crop + conv15_feat)
                else:
                    conv15_feat_crop = center_crop(conv15_feat, conv13_feat)
                    conv15_feat = self.relu(conv13_feat + conv15_feat_crop)
            else:
                conv15_feat = self.relu(conv13_feat + conv15_feat)

            # residual block
            conv16_feat = self.relu(self.norm_16(
                zero_conv2d(conv15_feat, self.resize16(conv16_weight[n]), bias=None, stride=1, padding=1)))
            conv17_feat = self.norm_17(
                zero_conv2d(conv16_feat, self.resize17(conv17_weight[n]), bias=None, stride=1, padding=1))
            if conv15_feat.shape != conv17_feat.shape:
                if conv15_feat.shape[2] > conv17_feat.shape[2] or conv15_feat.shape[3] > conv17_feat.shape[3]:
                    conv15_feat_crop = center_crop(conv15_feat, conv17_feat)
                    conv17_feat = self.relu(conv15_feat_crop + conv17_feat)
                else:
                    conv17_feat_crop = center_crop(conv17_feat, conv15_feat)
                    conv17_feat = self.relu(conv15_feat + conv17_feat_crop)
            else:
                conv17_feat = self.relu(conv15_feat + conv17_feat)

            # last 3 convs
            upsampled = F.interpolate(conv17_feat, scale_factor=2, mode='nearest')
            upsampled_padded = F.pad(upsampled, pad=(1,1,1,1), mode='reflect')
            conv18_feat = self.relu(
                self.norm_18(
                    F.conv2d(
                        upsampled_padded,
                        self.resize18(conv18_weight[n]),
                        bias=None,
                        stride=1,
                        padding=0
                    )
                )
            )
            conv19_feat = self.relu(self.norm_19(
                reflect_conv2d(conv18_feat, self.resize2(conv19_weight[n]), bias=None, stride=1, padding=1)))
            output = self.relu(
                reflect_conv2d(conv19_feat, self.resize3(conv20_weight[n]), bias=None, stride=1, padding=1))

            batch_outputs.append(output)

        output = torch.cat(batch_outputs, dim=0)
        return output

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
###### switch the trained network for differetn image operator  ######
netfile = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/models/model_train_wls_operator_SDR_nearestV2.net'
savepath = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/WLS-021544-predict_V4_realpng'

edgeCompute = EdgeComputation()

model = Model()
model.load_state_dict(torch.load(netfile))
model = model.cuda()
model.eval()

criterion = torch.nn.MSELoss(size_average=True)
criterion = criterion.cuda()


###### switch parameter values for different image operator here ######
# note all the parameter values will be aligned to the numerical range [0.002, 0.2]
# type = ['L0smooth']
# totalPara = [['000200', '000431', '002000', '009283', '020000']]
type = ['WLS']
totalPara = [['021544']]
# type = ['RTV']
# totalPara = [['000200', '000447', '001000', '002236', '005000']]
# type = ['RGF']
# totalPara = [['1.00000','3.25000','5.50000','7.75000','10.00000']]
# type = ['WMF']
# totalPara = [['1.00000','3.25000','5.50000','7.75000','10.00000']]
# type = ['shock']
# totalPara = [['1.00000']]
# type = ['deblock']
# totalPara = [['10','20']]
# type = ['rain']
# totalPara = [['10']]
# type = ['resolution']
# totalPara = [['2','3','4']]
# type = ['noise']
# totalPara = [['15','25','50']]

for m in range(len(type)):
    para = totalPara[m]
    loss_mse_ave = 0
    loss_psnr_ave = 0
    print(type[m])
    for n in range(len(para)):
        loss_mse = 0
        loss_psnr = 0
        loss_ssim = 0
        num = 0

        inputNameList, saveNameList, labelNameList = [], [], []
        if type[m] == 'L0smooth':
            print(float('%s.%s'%(para[n][:-5],para[n][-5:])))
            #parameter = float('%s.%s'%(para[n][:-5],para[n][-5:]))
            parameter = 0.25
            input_para = torch.zeros(1,1)
            input_para[0,0] = parameter
            inputPath = "./test_filter/"
            searchPath = "./test_filter/*-input.png"
            for inputName in glob.glob(searchPath):
                labelName = str.replace(inputName,'input','L0smooth-%s'%(para[n]))
                saveName = str.replace(inputName,'input','L0smooth-%s-predict'%(para[n]))
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)
        elif type[m] == 'WLS':
            print(float('%s.%s'%(para[n][:-5],para[n][-5:])))
            #parameter = float('%s.%s'%(para[n][:-5],para[n][-5:]))
            parameter = 0.25
            input_para = torch.zeros(1,1)
            input_para[0,0] = parameter/50
            #inputPath = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/input_V4"
            #### -- 生成大量用于 trainV9 的input数据 -- #####
            inputPath = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRinput"
            searchPath = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/input_V4/*-input.png"
            for inputName in glob.glob(searchPath):
                labelName = str.replace(inputName,'input','WLS-%s'%(para[n]))
                saveName = str.replace(inputName,'input','WLS-%s-predict'%(para[n]))
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)
        elif type[m] == 'RTV':
            print(float('%s.%s'%(para[n][:-5],para[n][-5:])))
            parameter = float('%s.%s'%(para[n][:-5],para[n][-5:]))
            input_para = torch.zeros(1,1)
            input_para[0,0] = (parameter-0.002)*4.125+0.002
            inputPath = "./test_filter/"
            searchPath = "./test_filter/*-input.png"
            for inputName in glob.glob(searchPath):
                labelName = str.replace(inputName,'input','RTV-%s'%(para[n]))
                saveName = str.replace(inputName,'input','RTV-%s-predict'%(para[n]))
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)
        elif type[m] == 'RGF':
            print(float('%s'%(para[n])))
            parameter = float('%s'%(para[n]))
            input_para = torch.zeros(1,1)
            input_para[0,0] = (parameter-1)*0.022+0.002
            inputPath = "./test_filter/"
            searchPath = "./test_filter/*-input.png"
            for inputName in glob.glob(searchPath):
                labelName = str.replace(inputName,'input','RGF-%s'%(para[n]))
                saveName = str.replace(inputName,'input','RGF-%s-predict'%(para[n]))
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)
        elif type[m] == 'WMF':
            print(float('%s'%(para[n])))
            parameter = float('%s'%(para[n]))
            input_para = torch.zeros(1,1)
            input_para[0,0] = (parameter-1)*0.022+0.002
            inputPath = "./test_filter/"
            searchPath = "./test_filter/*-input.png"
            for inputName in glob.glob(searchPath):
                labelName = str.replace(inputName,'input','WMF-%s'%(para[n]))
                saveName = str.replace(inputName,'input','WMF-%s-predict'%(para[n]))
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)
        elif type[m] == 'shock':
            print(float('%s'%(para[n])))
            parameter = float('%s'%(para[n]))
            input_para = torch.zeros(1,1)
            input_para[0,0] = 0.02
            inputPath = "./test_filter/"
            searchPath = "./test_filter/*-input.png"
            for inputName in glob.glob(searchPath):
                labelName = str.replace(inputName,'input','shock-%s'%(para[n]))
                saveName = str.replace(inputName,'input','shock-%s-predict'%(para[n]))
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)
        elif type[m] == 'deblock':
            print(float('%s'%(para[n])))
            parameter = float('%s'%(para[n]))            
            input_para = torch.zeros(1,1)
            input_para[0,0] = (parameter-10)*0.0198+0.002
            inputPath = "./test_deblock/LIVE1/"
            searchPath = "./test_deblock/LIVE1/*-label-y.png"
            for labelName in glob.glob(searchPath):
                inputName = str.replace(labelName,'-label-y.png','-deblock-y-%d.jpg'%(parameter))
                saveName = str.replace(labelName,'-label-y.png','-deblock-y-%d-predict.png'%(parameter))
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)
        elif type[m] == 'rain':
            inputPath = "./test_rain/"
            searchPath = "./test_rain/*_in.png"
            input_para = torch.zeros(1,1)
            input_para[0,0] = 0.02
            for inputName in glob.glob(searchPath):
                labelName = str.replace(inputName,'_in','_GT')
                saveName = str.replace(inputName,'_in','_predict')
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)
        elif type[m] == 'resolution':
            print(float('%s'%(para[n])))
            parameter = float('%s'%(para[n]))
            input_para = torch.zeros(1,1)
            input_para[0,0] = (parameter-2)*0.099+0.002
            inputPath = "./test_superresolution/"
            searchPath = "./test_superresolution/*-input.png"
            for labelName in glob.glob(searchPath):
                inputName = str.replace(labelName,'-input','-SR-y-%d'%(parameter))
                saveName = str.replace(labelName,'-input','-SR-y-%d-predict'%(parameter))
                labelName = str.replace(labelName,'-input','-input-y-%d'%(parameter))
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)
        elif type[m] == 'noise':
            print(float('%s'%(para[n])))
            parameter = float('%s'%(para[n]))
            input_para = torch.zeros(1,1)
            input_para[0,0] = (parameter-10)*0.0033+0.002
            inputPath = "./test_denoise/"
            searchPath = "./test_denoise/*-label-gray.png"
            for labelName in glob.glob(searchPath):
                inputName = str.replace(labelName,'-label-gray.png','-noise-gray-%d.png'%(parameter))
                saveName = str.replace(labelName,'-label-gray.png','-noise-gray-%d-predict.png'%(parameter))
                inputNameList.append(inputName)
                saveNameList.append(saveName)
                labelNameList.append(labelName)

        for i in range(len(inputNameList)):
            inputName = inputNameList[i]
            labelName = labelNameList[i]
            saveName = saveNameList[i]

            saveName_input = str.replace(inputName,inputPath,savepath)
            saveName = str.replace(saveName,inputPath,savepath)
            saveName_label = str.replace(labelName,inputPath,savepath)

            input = cv2.imread(inputName, cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(labelName, cv2.IMREAD_GRAYSCALE)
            input = input[:, :, np.newaxis]   # (500, 333, 1)
            label = label[:, :, np.newaxis]   # (500, 333, 1)

            input = input.transpose((2, 0, 1))
            label = label.transpose((2, 0, 1))
            input = torch.from_numpy(input)
            input = input.unsqueeze(0).float()
            label = torch.from_numpy(label)
            label = label.unsqueeze(0).float()
            input = input.cuda()
            
            inputs = torch.zeros(1,2,input.size(2),input.size(3))
            inputs[0,0:1] = input - 128
            inputs[0,1] = edgeCompute(input) - 128

            input_para = Variable(input_para)
            input_para = input_para.cuda()
            
            inputs = inputs.cuda()
            label = label.cuda()
            inputs = Variable(inputs)
            input = Variable(input)
            label = Variable(label, requires_grad=False)

            pred = model([input_para,inputs])

            min_h = min(pred.shape[2], label.shape[2])
            min_w = min(pred.shape[3], label.shape[3])
            pred = pred[:, :, :min_h, :min_w]
            label = label[:, :, :min_h, :min_w]
            loss = criterion(pred, label)
            loss.backward()

            # for j in range(3):
            #     numerator = torch.dot(pred[0,j].view(-1),input[0,j].view(-1))
            #     denominator = torch.dot(pred[0,j].view(-1),pred[0,j].view(-1))
            #     alpha = numerator/denominator
            #     pred[0,j] = pred[0,j] * alpha

            pred_numpy = pred.data[0].cpu().numpy()
            label_numpy = label.data[0].cpu().numpy()
            loss =  np.mean((pred_numpy - label_numpy)**2)
            loss_mse += loss
            loss_psnr += 10 *np.log10(255*255/loss)
            num += 1

            pred = pred.data[0].cpu()
            label = label.data[0].cpu()
            input = input.data[0].cpu()
            save_image(saveName,pred)
            #save_image(saveName_label,label)
            #save_image(saveName_input,input)

        print("mse: %f, psnr: %f "%(loss_mse/num, loss_psnr/num))
        loss_psnr_ave = loss_psnr_ave + loss_psnr/num
        loss_mse_ave = loss_mse_ave + loss_mse/num
    print("average mse: %f, psnr: %f"%(loss_mse_ave/len(para),loss_psnr_ave/len(para)))