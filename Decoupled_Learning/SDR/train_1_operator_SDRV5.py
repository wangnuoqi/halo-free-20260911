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

dtype = torch.cuda.FloatTensor
def load_tiff_with_log(input_path):
    input = tifffile.imread(input_path)
    #input = np.clip(input, 1e-6, None)
    if input is None:
        raise ValueError(f"Failed to read image at {input_path}")

    # 2. 灰度图处理成 [H, W, 1]
    if input.ndim == 2:
        img = np.expand_dims(input, axis=2)
    img = img.astype(np.float32)
    # img = np.nan_to_num(img, nan=1e-6, posinf=1e-6, neginf=1e-6)
    # img = np.clip(img, 1e-6, None)   
    #img = img / (img.max() + 1e-6)
    eps = 1e-7
    log_img = np.log(img + eps)
    min_val = log_img.min()
    max_val = log_img.max() # 等于 log(1)
    log_norm = (log_img - min_val) / (max_val - min_val + 1e-9)
    #output_path = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/123.tiff"
    #tifffile.imwrite(output_path, log_norm.astype(np.float32), photometric='minisblack')
    #print("Exists after write?", os.path.exists(output_path))
    #return log_norm
    return log_img 
def ensure_true_png(img_path):
    """
    检查img_path是否真是PNG格式。如果不是，则重新以PNG格式保存覆盖原文件，并返回新的路径。
    支持灰度和彩色图片。
    """
    # 先判断是否真的是PNG
    with open(img_path, 'rb') as f:
        header = f.read(8)
    is_png = header == b'\x89PNG\r\n\x1a\n'

    if is_png:
        return img_path  # 已经是PNG，无需处理

    # 否则重新读取（可选灰度/彩色），再以PNG格式重新保存
    try:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"无法读取文件: {img_path}")
        # 重新保存为真正的PNG
        cv2.imwrite(img_path, img)
        print(f"Converted {img_path} to true PNG format.")
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
    return img_path

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
            conv18_feat = self.relu(self.norm_18(
                F.conv_transpose2d(conv17_feat, self.resize4(conv18_weight[n]), bias=None, stride=2, padding=1)))
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

class EdgeComputation(nn.Module):
    def __init__(self):
        super(EdgeComputation, self).__init__()

    def forward(self, x):
        x_diffx = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])
        x_diffy = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])

        y = torch.Tensor(x.size())
        y.fill_(0)
        y[:,:,:,1:] += x_diffx
        y[:,:,:,:-1] += x_diffx
        y[:,:,1:,:] += x_diffy
        y[:,:,:-1,:] += x_diffy
        y = torch.sum(y,1)/3
        y /= 4
        return y

class RelativeMAELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RelativeMAELoss, self).__init__()
        self.eps = eps

    def forward(self, pred, label):
        # pred, label: [B, C, H, W] or [B, H, W]
        abs_error = torch.abs(pred - label)
        # denominator: 每张图片的最大值（可选：也可以用mean或其他统计量，看需求）
        # 取label在每张图（每个batch元素）上的绝对最大值
        denom = torch.amax(torch.abs(label), dim=[1,2,3], keepdim=True) + self.eps  # shape: [B,1,1,1]
        rel_error = abs_error / denom
        # 返回batch内所有像素的平均相对误差
        return rel_error.mean()

class RelativeMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RelativeMSELoss, self).__init__()
        self.eps = eps

    def forward(self, pred, label):
        # pred, label: [B, C, H, W] or [B, H, W]
        squared_error = (pred - label) ** 2
        # denominator: 每张图片的最大label
        denom = torch.amax(torch.abs(label), dim=[1,2,3], keepdim=True) + self.eps
        rel_sq_error = squared_error / (denom ** 2)  # 注意MSE归一化一般分母要平方
        return rel_sq_error.mean()

def var_custom_collate(batch):
    min_h, min_w = 10000, 10000
    for item in batch:
        min_h = min(min_h, item['input_height'])
        min_w = min(min_w, item['input_width'])

    batch_input_images = torch.Tensor(len(batch), 2, min_h, min_w)
    batch_target_images = torch.Tensor(len(batch), 1, min_h, min_w)
    batch_input_params = torch.Tensor(len(batch), 1)
    for idx, item in enumerate(batch):
        off_y = random.randint(0, item['input_height']-min_h)
        off_x = random.randint(0, item['input_width']-min_w)
        batch_input_images[idx,0:1] = item['input_img'][:, :, off_y:off_y+min_h, off_x:off_x+min_w]
        #batch_input_images[idx,1] = edgeCompute(batch_input_images.narrow(0,idx,1).narrow(1,0,1))
        batch_input_images[idx,1] = item['input_edge'][:, :, off_y:off_y+min_h, off_x:off_x+min_w]
        batch_target_images[idx] = item['target_img'][:, :, off_y:off_y+min_h, off_x:off_x+min_w]
        batch_input_params[idx, 0] = item['input_param']
        # batch_input_params[idx, 1] = item['input_type']

    return (batch_input_images, batch_input_params, batch_target_images)

#######(2) data #####################
class ArbitraryImageFolder(data.Dataset):

    def __init__(self, input_paths, label_paths, params, type, hws, input_edge_paths):
        self.input_paths = input_paths
        self.label_paths = label_paths
        self.params = params
        self.type= type
        self.hws = hws
        self.input_edge_paths = input_edge_paths
        self.size = len(input_paths)

    def __getitem__(self, index):
        input_path, label_path, para, type, input_edge_paths = self.input_paths[index], self.label_paths[index], self.params[index], self.type[index], self.input_edge_paths[index]
        img_h, img_w = self.hws[index]

        # input_path = ensure_true_png(input_path)
        # label_path = ensure_true_png(label_path)
        input = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        input_edge = cv2.imread(input_edge_paths, cv2.IMREAD_GRAYSCALE)

        input = input[:, :, np.newaxis]   # (500, 333, 1)
        input_edge = input_edge[:, :, np.newaxis]
        label = label[:, :, np.newaxis]   # (500, 333, 1)
        input = input.transpose((2, 0, 1))
        input = torch.from_numpy(input).unsqueeze(0).float()
        input_edge = input_edge.transpose((2, 0, 1))
        input_edge = torch.from_numpy(input_edge).unsqueeze(0).float()
        label = label.transpose((2, 0, 1))
        label = torch.from_numpy(label).unsqueeze(0).float()

        return {'input_img': input, 'target_img': label, 'input_param': para, 'input_type': type, 'input_height': img_h, 'input_width': img_w, 'input_edge': input_edge}

    def __len__(self):
        return self.size



##### change the saved model filename for another operator##### 
name = 'train_wls_operator_SDR_edge'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
epoches = 200
batchsize = 12
num_workers = 12

edgeCompute = EdgeComputation()

# with open('/mnt/codes/imageOperator_codes/train_1_operator.py', 'r') as fin:
#     print(fin.read())
print("***********************************************")

##### filter_input_folder contains the orignial natural images in the dataset #####
filter_input_folder = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRinput'
filter_edge_input_folder = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRinput_edge_label'
filter_input_folder_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRtest'
filter_edge_input_folder_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRtest_edge'
##### the following folders contains the corrupted images taken as input in the restoration tasks #####
# rain_input_folder = '/mnt/data/sky_rainy_dataset/ground_truth_even/'
# sr_input_folder = '/mnt/data/VOC2012_srgray234_input_various_uniform_hw/'
# noise_input_folder = '/mnt/data/VOC2012_denoisegray_input_various_uniform_hw/'
# deblock_input_folder = '/mnt/data/VOC2012_deblock_input_various_uniform_hw/'


#####(1) uncomment corresponding codes for different image operators ##### 
train_input_paths = []
train_label_paths = []
train_input_edge_paths = []
train_params = []
train_type = []
train_hws = []
with open("/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR_wls_SDRretrain512_list.txt","r") as f:
    for line in f:
        label_path = line.strip()
        img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        img_h, img_w = img.shape[0], img.shape[1]
        label_imgname = os.path.splitext(os.path.basename(label_path))[0]
        name_parts = label_imgname.split('-')
        #img_h, img_w = 512, 512      
        #type = label_path.split('/')[3].split('_')[1]
        type = 'WLS'
        # if type == 'L0smooth':
        #     param = float(name_parts[-3])
        #     train_type.append(float('0.1'))
        #     train_params.append(param)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     train_label_paths.append(label_path)
        if type == 'WLS':
            #param = float(name_parts[-3])
            param = 0.25
            train_type.append(float('0.2'))
            train_params.append(param/50)
            base_imgname = label_imgname.replace('-%s-%s'%(name_parts[-2], name_parts[-1]), '')
            edge_imgname = label_imgname.replace('-%s'%(name_parts[-3]), '')
            train_input_paths.append(os.path.join(filter_input_folder, base_imgname + '.png'))
            train_input_edge_paths.append(os.path.join(filter_edge_input_folder, edge_imgname + '.png'))
            train_label_paths.append(label_path)
        # if type == 'RTV':
        #     param = float(name_parts[-3])
        #     train_type.append(float('0.3'))
        #     train_params.append((param-0.002)*4.125+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     train_label_paths.append(label_path)
        # if type == 'RGF':
        #     param = float(name_parts[-3])
        #     train_type.append(float('0.4'))
        #     train_params.append((param-1)*0.022+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     train_label_paths.append(label_path)
        # if type == 'WMF':
        #     param = float(name_parts[-3])
        #     train_type.append(float('0.5'))
        #     train_params.append((param-1)*0.022+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     train_label_paths.append(label_path)
        # if type == 'shock':
        #     param = float(name_parts[-3])
        #     train_type.append(float('0.6'))
        #     train_params.append(0.02)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     train_label_paths.append(label_path)
        # if type == 'fastLLFenhancement':
        #     param = float(name_parts[-3])
        #     train_type.append(float('0.6'))
        #     train_params.append((param-2)*0.033+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     train_label_paths.append(label_path)
        # if type == 'deblock':
        #     param = float(name_parts[-3])
        #     train_type.append(float('0.7'))
        #     train_params.append((param-10)*0.0198+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     train_input_paths.append(label_path)
        #     train_label_paths.append(os.path.join(deblock_input_folder, base_imgname+'.png'))
        # if type == 'rainy':
        #     train_type.append(float('0.8'))
        #     train_params.append(0.02)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     train_input_paths.append(label_path)
        #     train_label_paths.append(os.path.join(rain_input_folder, base_imgname+'.png'))
        # if type == 'srgray234':
        #     param = float(name_parts[-3])
        #     train_type.append(float('0.9'))
        #     train_params.append((param-2)*0.099+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s'%(name_parts[-2], name_parts[-1]), '')
        #     train_label_paths.append(os.path.join(sr_input_folder, base_imgname+'.png'))
        #     train_input_paths.append(label_path)
        # if type == 'denoisegray':
        #     param = float(name_parts[-3])
        #     train_type.append(float('1.0'))
        #     train_params.append((param-10)*0.0033+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     train_label_paths.append(os.path.join(noise_input_folder, base_imgname+'.png'))
        #     train_input_paths.append(label_path)
        else:
            continue
        
        train_hws.append((img_h, img_w))

test_input_paths = []
test_label_paths = []
test_input_edge_paths = []
test_params = []
test_type = []
test_hws = []
with open("vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR_wls_SDRretest512_list.txt","r") as f:
    for line in f:
        label_path = line.strip()
        img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        img_h, img_w = img.shape[0], img.shape[1]
        label_imgname = os.path.splitext(os.path.basename(label_path))[0]
        name_parts = label_imgname.split('-')
        #img_h, img_w = 512, 512
        type = 'WLS'
        # if type == 'L0smooth':
        #     param = float(name_parts[-3])
        #     test_type.append(float('0.1'))
        #     test_params.append(param)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     test_label_paths.append(label_path)
        if type == 'WLS':
            param = 0.25
            test_type.append(float('0.2'))
            test_params.append(param/50)
            base_imgname = label_imgname.replace('-%s-%s'%(name_parts[-2], name_parts[-1]), '')
            edge_imgname = label_imgname.replace('-%s'%(name_parts[-3]), '')
            test_input_paths.append(os.path.join(filter_input_folder_test, base_imgname+'.png'))
            test_input_edge_paths.append(os.path.join(filter_edge_input_folder_test, edge_imgname + '.png'))
            test_label_paths.append(label_path)
        # if type == 'RTV':
        #     param = float(name_parts[-3])
        #     test_type.append(float('0.3'))
        #     test_params.append((param-0.002)*4.125+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     test_label_paths.append(label_path)
        # if type == 'RGF':
        #     param = float(name_parts[-3])
        #     test_type.append(float('0.4'))
        #     test_params.append((param-1)*0.022+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     test_label_paths.append(label_path)
        # if type == 'WMF':
        #     param = float(name_parts[-3])
        #     test_type.append(float('0.5'))
        #     test_params.append((param-1)*0.022+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     test_label_paths.append(label_path)
        # if type == 'shock':
        #     param = float(name_parts[-3])
        #     test_type.append(float('0.6'))
        #     test_params.append(0.02)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     test_label_paths.append(label_path)
        # if type == 'fastLLFenhancement':
        #     param = float(name_parts[-3])
        #     test_type.append(float('0.6'))
        #     test_params.append((param-2)*0.033+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     test_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.png'))
        #     test_label_paths.append(label_path)
        # if type == 'deblock':
        #     param = float(name_parts[-3])
        #     test_type.append(float('0.7'))
        #     test_params.append((param-10)*0.0198+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     test_input_paths.append(label_path)
        #     test_label_paths.append(os.path.join(deblock_input_folder, base_imgname+'.png'))
        # if type == 'rainy':
        #     test_type.append(float('0.8'))
        #     test_params.append(0.02)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     test_input_paths.append(label_path)
        #     test_label_paths.append(os.path.join(rain_input_folder, base_imgname+'.png'))
        # if type == 'srgray234':
        #     param = float(name_parts[-3])
        #     test_type.append(float('0.9'))
        #     test_params.append((param-2)*0.099+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s'%(name_parts[-2], name_parts[-1]), '')
        #     test_label_paths.append(os.path.join(sr_input_folder, base_imgname+'.png'))
        #     test_input_paths.append(label_path)
        # if type == 'denoisegray':
        #     param = float(name_parts[-3])
        #     test_type.append(float('1.0'))
        #     test_params.append((param-10)*0.0033+0.002)
        #     base_imgname = label_imgname.replace('_%s_%s_%s'%(name_parts[-3], name_parts[-2], name_parts[-1]), '')
        #     test_label_paths.append(os.path.join(noise_input_folder, base_imgname+'.png'))
        #     test_input_paths.append(label_path)
        else:
            continue

        test_hws.append((img_h, img_w))

trainSetSize, testSetSize = len(train_input_paths), len(test_input_paths)
epochSize = 279

model = Model()
criterion = torch.nn.MSELoss(size_average=True)
criterion_mae = RelativeMAELoss()
criterion_mse = RelativeMSELoss()

model = torch.nn.DataParallel(model.cuda())

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_dataset = ArbitraryImageFolder(train_input_paths, train_label_paths, train_params, train_type, train_hws, train_input_edge_paths)
train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, collate_fn=var_custom_collate,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

test_dataset = ArbitraryImageFolder(test_input_paths, test_label_paths, test_params, test_type, test_hws, test_input_edge_paths)
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
    # if p == 40:
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # if p == 100:
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    if p == 80:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for curr_iter, batch_data in enumerate(train_dataloader):
        iter += 1

        inputTensor, inputParaTensor, labelTensor = batch_data
        inputTensor = inputTensor - 128
        inputTensor_v = Variable(inputTensor.cuda())
        inputParaTensor_v = Variable(inputParaTensor.cuda())
        labelTensor_v = Variable(labelTensor.cuda())

        optimizer.zero_grad()
        pred_v = model([inputParaTensor_v, inputTensor_v])

        min_h = min(pred_v.shape[2], labelTensor_v.shape[2])
        min_w = min(pred_v.shape[3], labelTensor_v.shape[3])
        pred_v = pred_v[:, :, :min_h, :min_w]
        labelTensor_v = labelTensor_v[:, :, :min_h, :min_w]

        loss = criterion(pred_v, labelTensor_v)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_count += 1
        writer.add_scalar('Loss/train_batch', loss.item(), iter)
        print('Iter: %d Current loss: %.4f'%(iter,loss.item()))

        if iter % int(epochSize/batchsize) == 0:
            break
    writer.add_scalar('Loss/train_epoch_avg', train_loss / train_count, p)

    for curr_iter, batch_data in enumerate(test_dataloader):
        iterTest += 1

        inputTensor, inputParaTensor, labelTensor = batch_data
        inputTensor = inputTensor - 128
        inputTensor_v = Variable(inputTensor.cuda())
        inputParaTensor_v = Variable(inputParaTensor.cuda())
        labelTensor_v = Variable(labelTensor.cuda())

        pred_v = model([inputParaTensor_v, inputTensor_v])
        min_h = min(pred_v.shape[2], labelTensor_v.shape[2])
        min_w = min(pred_v.shape[3], labelTensor_v.shape[3])
        pred_v = pred_v[:, :, :min_h, :min_w]
        labelTensor_v = labelTensor_v[:, :, :min_h, :min_w]
        loss = criterion(pred_v, labelTensor_v)
        # loss.backward()

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