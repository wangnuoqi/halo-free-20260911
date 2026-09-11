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

def save_image(filename, data):
    # data: numpy array or torch tensor, shape [C, H, W] or [1, H, W] or [3, H, W] or [B, C, H, W]
    # 1. 去batch维
    if data.ndim == 4:
        data = data[0]  # 默认取第一个batch
    # 2. 保证数据在CPU和numpy类型
    if hasattr(data, 'detach'):
        data = data.detach().cpu().numpy()
    # 3. squeeze通道（如果是1通道）
    if data.shape[0] == 1:
        img = data[0]   # [H, W]
        img = np.clip(img, 0, 255)
        img = img.astype("uint8")
        cv2.imwrite(filename, img)
    elif data.shape[0] == 3:
        img = data.transpose(1, 2, 0)  # [H, W, 3]
        img = np.clip(img, 0, 255)
        img = img.astype("uint8")
        cv2.imwrite(filename, img)
    else:
        # 其他通道数：自动转为灰度
        img = data[0]
        img = np.clip(img, 0, 255)
        img = img.astype("uint8")
        cv2.imwrite(filename, img)

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(x + out)

class IntrinsicNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        # 前3层（第1层stride=2下采样）
        self.conv1 = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 10个残差块（20层dilated conv）
        self.resblocks = nn.Sequential(*[
            ResidualBlock(64, dilation=2) for _ in range(10)
        ])

        # 后3层（上采样回原分辨率，最后一层不BN/ReLU）
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 恢复到输入大小

    def forward(self, x):
        # x: [B,1,H,W]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.resblocks(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = self.upsample(x)
        return x

# --- 2. Guidance Network (18层, 只接收单通道输入) ---
class GuidanceNet(nn.Module):
    def __init__(self):
        super(GuidanceNet, self).__init__()

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
        # x_param = x_param.to(next(self.parameters()).device)
        # x_img = x_img.to(next(self.parameters()).device)

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
            # first 3 convs
            conv1_feat = self.relu(self.norm_1(
                F.conv2d(x_img.narrow(0, n, 1), self.resize1(conv1_weight[n]), bias=None, padding=1)))
            conv2_feat = self.relu(self.norm_2(
                F.conv2d(conv1_feat, self.resize2(conv2_weight[n]), bias=None, padding=1)))
            conv3_feat = self.relu(self.norm_3(
                F.conv2d(conv2_feat,self.resize2(conv3_weight[n]), bias=None, stride=2, padding=1)))

            # residual block
            conv4_feat = self.relu(self.norm_4(
                F.conv2d(conv3_feat, self.resize2(conv4_weight[n]), bias=None, stride=1, padding=2, dilation=2)))
            conv5_feat = self.norm_5(
                F.conv2d(conv4_feat, self.resize2(conv5_weight[n]), bias=None, stride=1, padding=2, dilation=2))
            conv5_feat = self.relu(conv3_feat + conv5_feat)

            # residual block
            conv6_feat = self.relu(self.norm_6(
                F.conv2d(conv5_feat, self.resize2(conv6_weight[n]), bias=None, stride=1, padding=4, dilation=4)))
            conv7_feat = self.norm_7(
                F.conv2d(conv6_feat, self.resize2(conv7_weight[n]), bias=None, stride=1, padding=4, dilation=4))
            conv7_feat = self.relu(conv5_feat + conv7_feat)

            # residual block
            conv8_feat = self.relu(self.norm_8(
                F.conv2d(conv7_feat, self.resize2(conv8_weight[n]), bias=None, stride=1, padding=4, dilation=4)))
            conv9_feat = self.norm_9(
                F.conv2d(conv8_feat, self.resize2(conv9_weight[n]), bias=None, stride=1, padding=4, dilation=4))
            conv9_feat = self.relu(conv7_feat + conv9_feat)

            # residual block
            conv10_feat = self.relu(self.norm_10(
                F.conv2d(conv9_feat, self.resize2(conv10_weight[n]), bias=None, stride=1, padding=8, dilation=8)))
            conv11_feat = self.norm_11(
                F.conv2d(conv10_feat, self.resize2(conv11_weight[n]), bias=None, stride=1, padding=8, dilation=8))
            conv11_feat = self.relu(conv9_feat + conv11_feat)

            # residual block
            conv12_feat = self.relu(self.norm_12(
                F.conv2d(conv11_feat, self.resize2(conv12_weight[n]), bias=None, stride=1, padding=8, dilation=8)))
            conv13_feat = self.norm_13(
                F.conv2d(conv12_feat, self.resize2(conv13_weight[n]), bias=None, stride=1, padding=8, dilation=8))
            conv13_feat = self.relu(conv11_feat + conv13_feat)

            # residual block
            conv14_feat = self.relu(self.norm_14(
                F.conv2d(conv13_feat, self.resize2(conv14_weight[n]), bias=None, stride=1, padding=16, dilation=16)))
            conv15_feat = self.norm_15(
                F.conv2d(conv14_feat, self.resize2(conv15_weight[n]), bias=None, stride=1, padding=16, dilation=16))
            conv15_feat = self.relu(conv13_feat + conv15_feat)

            # residual block
            conv16_feat = self.relu(self.norm_16(
                F.conv2d(conv15_feat, self.resize2(conv16_weight[n]), bias=None, stride=1, padding=1, dilation=1)))
            conv17_feat = self.norm_17(
                F.conv2d(conv16_feat, self.resize2(conv17_weight[n]), bias=None, stride=1, padding=1, dilation=1))
            conv17_feat = self.relu(conv15_feat + conv17_feat)

            # last 3 convs
            upsampled = F.interpolate(conv17_feat, scale_factor=2, mode='nearest')
            upsampled_padded = F.pad(upsampled, pad=(1,1,1,1), mode='replicate')
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
            conv19_feat = self.relu(self.norm_19(
                F.conv2d(conv18_feat, self.resize2(conv19_weight[n]), bias=None, stride=1, padding=1)))
            output = self.relu(F.conv2d(conv19_feat, self.resize3(conv20_weight[n]), bias=None, stride=1, padding=1))

            batch_outputs.append(output)

        output = torch.cat(batch_outputs, dim=0)
        return output

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

def recursive_filter_row(x_row, e_row, eps=1e-2):
    # x_row, e_row: [W]
    y_list = [x_row[0:1]]
    y_prev = x_row[0:1]
    for w in range(1, x_row.shape[0]):
        g = torch.exp(-torch.abs(e_row[w] - e_row[w-1]) / eps)
        new_y = (1-g) * x_row[w:w+1] + g * y_prev
        y_list.append(new_y)
        y_prev = new_y
    return torch.cat(y_list, dim=0)
from torch.func import vmap
def domain_filter(X, E, eps=1e-2):
    """
    X: [B, 1, H, W]
    E: [B, 1, H, W]
    """
    B, C, H, W = X.shape
    assert C == 1, "Only support single channel"
    
    # 横向 vmap (batch和每一行)
    x_rows = X[:, 0, :, :]  # [B, H, W]
    e_rows = E[:, 0, :, :]  # [B, H, W]
    x_rows_flat = x_rows.reshape(-1, W)   # [B*H, W]
    e_rows_flat = e_rows.reshape(-1, W)   # [B*H, W]

    # 先vmap一层
    filter_row_fn = lambda x, e: recursive_filter_row(x, e, eps)
    Y_rows_flat = vmap(filter_row_fn)(x_rows_flat, e_rows_flat)  # [B*H, W]
    Y_rows = Y_rows_flat.reshape(B, H, W)  # [B, H, W]

    # 纵向 vmap（对每列）
    # 转置 [B, H, W] -> [B, W, H]
    Y_rows_t = Y_rows.permute(0, 2, 1).contiguous()  # [B, W, H]
    E_cols_t = e_rows.permute(0, 2, 1).contiguous()  # [B, W, H]
    Y_cols_flat = Y_rows_t.reshape(-1, H)  # [B*W, H]
    E_cols_flat = E_cols_t.reshape(-1, H)  # [B*W, H]

    Y_cols_flat = vmap(filter_row_fn)(Y_cols_flat, E_cols_flat)  # [B*W, H]
    Y_cols = Y_cols_flat.reshape(B, W, H).permute(0, 2, 1).contiguous()  # [B, H, W]

    # 加回通道维
    Y_out = Y_cols.unsqueeze(1)  # [B, 1, H, W]
    return Y_out

# --- 4. 整体网络结构 ---
class EdgePreservingNet(nn.Module):
    def __init__(self, eps=1e-2):
        super(EdgePreservingNet, self).__init__()
        self.guidance_net = GuidanceNet()
        self.intrinsic_net = IntrinsicNet(in_ch=1, out_ch=1)
        self.eps = eps

    def forward(self, guidance_input, input_img):
        # x: [B, 1, H, W] 单通道
        R_coarse = self.intrinsic_net(input_img)
        edge_map = self.guidance_net(guidance_input)                    # [B,2,H,W]
 
        # min_h = min(edge_map.shape[2], net_input.shape[2])
        # min_w = min(edge_map.shape[3], net_input.shape[3])
        # edge_map = edge_map[:, :, :min_h, :min_w]
        # net_input = net_input[:, :, :min_h, :min_w]

        #smoothed = domain_filter(net_input, edge_map, eps=self.eps) # [B,1,H,W]
        return R_coarse, edge_map

# # --- 5. Loss函数示例 ---
# def compute_loss(smoothed, label, edge_map, edge_gt, lambda_edge=0.5):
#     # smoothed: pred, label:目标, edge_map:生成边缘, edge_gt:真实边缘
#     loss1 = F.mse_loss(edge_map, edge_gt)
#     loss2 = F.mse_loss(smoothed, label)
#     return lambda_edge * loss1 + loss2

def var_custom_collate(batch):
    min_h, min_w = 10000, 10000
    for item in batch:
        min_h = min(min_h, item['input_height'])
        min_w = min(min_w, item['input_width'])

    batch_size = len(batch)
    # 1. guidance_net 输入：guidance_input + guidance_edge
    batch_guidance_input = torch.zeros(batch_size, 2, min_h, min_w)
    # 2. main net 输入
    batch_net_input = torch.zeros(batch_size, 1, min_h, min_w)
    # 3. guidance_label
    batch_net_label = torch.zeros(batch_size, 1, min_h, min_w)
    #
    batch_guidance_label = torch.zeros(batch_size, 1, min_h, min_w)
    # 5. param
    batch_input_params = torch.zeros(batch_size, 1)

    for idx, item in enumerate(batch):
        off_y = random.randint(0, item['input_height'] - min_h)
        off_x = random.randint(0, item['input_width'] - min_w)

        # 拼接 guidance_input 和 guidance_edge -> [2, H, W]
        batch_guidance_input[idx, 0] = item['guidance_input'][:, off_y:off_y+min_h, off_x:off_x+min_w]
        batch_guidance_input[idx, 1] = item['guidance_edge'][:, off_y:off_y+min_h, off_x:off_x+min_w]

        # main net 输入
        batch_net_input[idx, 0] = item['net_input'][:, off_y:off_y+min_h, off_x:off_x+min_w]
        # guidance label
        batch_guidance_label[idx, 0] = item['guidance_label'][:, off_y:off_y+min_h, off_x:off_x+min_w]
        # net label
        batch_net_label[idx, 0] = item['net_label'][:, off_y:off_y+min_h, off_x:off_x+min_w]
        # param
        batch_input_params[idx, 0] = item['input_param']

    # 返回顺序：guidance_net输入, main net输入, param, guidance_label, net_label
    return (
        batch_guidance_input,    # [B, 2, H, W]
        batch_net_input,         # [B, 1, H, W]
        batch_input_params,      # [B, 1]
        batch_guidance_label,    # [B, 1, H, W]
        batch_net_label          # [B, 1, H, W]
    )

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

guidance_input_folder_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRtest'
guidance_edge_input_folder_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_edge_V7_test_input'
guidance_label_folder_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_edge_V7_test_label'

net_input_folder_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRtest'
net_label_folder_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRtest_label'

SAVE_FOLDER = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/eva_V8'
SAVE_FOLDER_EDGE = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/eva_V8_edge'
SAVE_FOLDER_ORIGIN = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/eva_V8_origin'

test_guidance_input_paths = []
test_guidance_input_edge_paths = []
test_net_input_paths = []

test_gudiance_label_paths = []
test_net_label_paths = []

test_params = []
test_type = []
test_hws = []

saveNameList = []
saveNameList_edge = []
saveNameList_origin = [] 
with open("vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR_wls_SDRretest512_list.txt","r") as f:
    for line in f:
        label_path = line.strip()
        img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        img_h, img_w = img.shape[0], img.shape[1]
        label_imgname = os.path.splitext(os.path.basename(label_path))[0]
        name_parts = label_imgname.split('-')
        #img_h, img_w = 512, 512
        type = 'WLS'
        if type == 'WLS':
            #param = float(name_parts[-3])
            param = 0.25
            test_type.append(float('0.2'))
            test_params.append(param/50)
            ###base_imgname: 2012_004179-input  edge_imgname 2012_004179-WLS-021544
            base_imgname = label_imgname.replace('-%s-%s'%(name_parts[-2], name_parts[-1]), '')
            edge_imgname = label_imgname.replace('-%s'%(name_parts[-3]), '')

            test_net_input_paths.append(os.path.join(net_input_folder_test, base_imgname + '.png'))
            test_guidance_input_paths.append(os.path.join(net_input_folder_test, base_imgname + '.png'))
            test_guidance_input_edge_paths.append(os.path.join(guidance_edge_input_folder_test, base_imgname + '_edge.png' ))

            test_net_label_paths.append(label_path)
            test_gudiance_label_paths.append(os.path.join(guidance_label_folder_test, label_imgname + '_edge.png'))
            saveName = os.path.join(edge_imgname + '_pred.png')
            saveName_edge = os.path.join(edge_imgname + '_edge_pred.png')
            saveName_origin = os.path.join(edge_imgname + '_origin_pred.png')

            saveNameList.append(os.path.join(SAVE_FOLDER, saveName))
            saveNameList_edge.append(os.path.join(SAVE_FOLDER_EDGE, saveName_edge))
            saveNameList_origin.append(os.path.join(SAVE_FOLDER_ORIGIN, saveName_origin))
        else:
            continue
        
        test_hws.append((img_h, img_w))

model = EdgePreservingNet()
model_path = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/models/model_train_edge.net' 
model.load_state_dict(torch.load(model_path))
model = model.cuda()
model.eval()
batchsize = 1
num_workers = 12
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

test_dataset = ArbitraryImageFolder(test_guidance_input_paths, test_guidance_input_edge_paths, test_net_input_paths, test_gudiance_label_paths, test_net_label_paths,
                            test_params, test_type, test_hws)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, collate_fn=var_custom_collate,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

criterion = torch.nn.MSELoss(reduction='mean')
total_loss = 0
total_psnr = 0
iterTest = 0
num = 0

with torch.no_grad():
    for i, batch_data in enumerate(test_dataloader):
        guidance_input_Tensor, net_input_Tensor, inputParaTensor, guidance_label_Tensor, net_label_Tensor = batch_data
        
        guidance_input_Tensor = guidance_input_Tensor - 128
        net_input_Tensor = net_input_Tensor - 128

        guidance_input_Tensor_v = Variable(guidance_input_Tensor.cuda())
        net_input_Tensor_v = Variable(net_input_Tensor.cuda())
        inputParaTensor_v = Variable(inputParaTensor.cuda())
        guidance_label_Tensor_v = Variable(guidance_label_Tensor.cuda())
        net_label_Tensor_v = Variable(net_label_Tensor.cuda())

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

        total_loss += loss.item()
        num += 1
    ###### -- 将数据转移到cpu上 --- #########
        smoothed_np = smoothed.cpu().numpy()
        R_coarse_np = R_coarse.cpu().numpy()
        edge_map_np = edge_map.cpu().numpy()
        
        net_label_Tensor_v_np = net_label_Tensor_v.cpu().numpy()
        mse = np.mean((smoothed_np - net_label_Tensor_v_np) ** 2)
        if mse == 0:
            psnr = 100  # 理论最大
        else:
            psnr = 10 * np.log10(255 * 255 / mse)
        total_psnr += psnr
        #raw_name = os.path.basename(test_input_paths[batch_idx])
        save_dir_output = "vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/WLS-021544-predict_V4"
        save_dir_edge = "vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/WLS-021544-predict_V4_edge"
        save_dir_output1 = "vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/WLS-021544-predict_V4_output1"

        save_path = saveNameList[i]
        save_path_edge = saveNameList_edge[i]
        save_path_origin = saveNameList_origin[i]

        save_image(save_path, smoothed_np)
        save_image(save_path_edge, edge_map_np)
        save_image(save_path_origin, R_coarse_np)
        
        print(f"[{i}], loss: {loss_net.item():.6f}, psnr: {psnr:.2f} dB")

avg_loss = total_loss / num
avg_psnr = total_psnr / num
print(f"Average test loss: {avg_loss:.6f}, Average PSNR: {avg_psnr:.2f} dB")

