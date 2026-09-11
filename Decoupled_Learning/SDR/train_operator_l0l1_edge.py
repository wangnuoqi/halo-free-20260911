import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.data as data
import math
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
# import torchvision.ops
# from torchvision.ops import DeformConv2d

import imagecodecs
import cv2
import scipy.misc
import os
from skimage import io
from os import listdir
from os.path import join
from PIL import Image
import time
import tifffile
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(64, affine=True)

        self.resblocks = nn.Sequential(
            ResidualBlock(64, dilation=2),
            ResidualBlock(64, dilation=4),
            ResidualBlock(64, dilation=8),
            ResidualBlock(64, dilation=16),
            ResidualBlock(64, dilation=1)
        )

        self.up = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.norm_up = nn.InstanceNorm2d(64, affine=True)

        self.conv_out1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.norm_out1 = nn.InstanceNorm2d(64, affine=True)
        self.conv_out2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x_input):
        B, _, H, W = x_input.shape
        x = torch.cat([x_input], dim=1)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.relu(self.norm3(self.conv3(x)))

        x = self.resblocks(x)

        x = self.relu(self.norm_up(self.up(x)))
        x = self.relu(self.norm_out1(self.conv_out1(x)))
        x = self.conv_out2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + residual)

def load_tiff_with_log(input_path):
    input = tifffile.imread(input_path)
    input = np.clip(input, 1e-6, None)
    if input is None:
        raise ValueError(f"Failed to read image at {input_path}")

    # 2. 灰度图处理成 [H, W, 1]
    if input.ndim == 2:
        img = np.expand_dims(input, axis=2)
    img = img.astype(np.float32)
    img = np.nan_to_num(img, nan=1e-6, posinf=1e-6, neginf=1e-6)
    img = np.clip(img, 1e-6, None)   
    img = img / (img.max() + 1e-6)
    log_img = np.log(img)
    min_val = log_img.min()
    max_val = 0.0  # 等于 log(1)
    log_norm = (log_img - min_val) / (max_val - min_val + 1e-6)
    #output_path = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/123.tiff"
    #tifffile.imwrite(output_path, log_norm.astype(np.float32), photometric='minisblack')
    #print("Exists after write?", os.path.exists(output_path))
    return log_norm

######图像边缘计算 更换为直接提取边缘数据用于输入###########   
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

        # y_np = y.detach().cpu().numpy().astype(np.float32)
        # output_path = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/123.tiff"
        # tifffile.imwrite(output_path, y_np.astype(np.float32), photometric='minisblack')
        # print("Exists after write?", os.path.exists(output_path))
        return y

class EdgeComputation2(nn.Module):
    def __init__(self):
        super(EdgeComputation2, self).__init__()

    def forward(self, input_path, edge_root, edge_suffix):
    # 1. 提取 input_path 中的图像 ID（不含扩展名）
        input_name = os.path.splitext(os.path.basename(input_path))[0]

        # 2. 构造 edge 路径
        edge_filename = f"{input_name}{edge_suffix}"
        edge_path = os.path.join(edge_root, edge_filename)

        # 3. 读取边缘图
        edge_img = tifffile.imread(edge_path)  # shape: [H, W]
        if edge_img is None:
            raise FileNotFoundError(f"[Edge Missing] Cannot find edge file: {edge_path}")
        
        # 4. 转为 tensor，归一化 & 加通道
        edge_tensor = torch.from_numpy(edge_img).float()     # [H, W] → float32
        edge_tensor = edge_tensor.unsqueeze(0)                       # → [1, H, W]
        return edge_tensor


def var_custom_collate(batch):
    min_h, min_w = 10000, 10000
    for item in batch:
        min_h = min(min_h, item['input_height'])
        min_w = min(min_w, item['input_width'])

    batch_input_images = torch.Tensor(len(batch), 2, min_h, min_w)
    batch_target_images = torch.Tensor(len(batch), 1, min_h, min_w)
    #batch_edge = torch.Tensor(len(batch), 1, min_h, min_w)
    #batch_input_params = torch.Tensor(len(batch), 1)
    for idx, item in enumerate(batch):
        off_y = random.randint(0, item['input_height']-min_h)
        off_x = random.randint(0, item['input_width']-min_w)
        batch_input_images[idx,0] = item['input_img'][:, :, off_y:off_y+min_h, off_x:off_x+min_w]
        batch_target_images[idx] = item['target_img'][:, :, off_y:off_y+min_h, off_x:off_x+min_w]
#########将原本需要计算的边缘直接输入batch通道内###########
        # batch_input_images[idx,3] = edgeCompute(input_path,
        #         edge_root='/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR512_l0l1_label_0.3_edge',
        #         edge_suffix='_l1l0_edge.png')
        batch_input_images[idx,1] = edgeCompute(batch_target_images.narrow(0,idx,1).narrow(1,0,1))
        #batch_input_images[idx,1] = edgeCompute2(item['input_path'],
        #         edge_root='/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR512_l0l1_label_edgeV2_train_test',
        #         edge_suffix='_l1l0.tiff')
        #batch_target_images[idx] = item['target_img'][:, :, off_y:off_y+min_h, off_x:off_x+min_w]
        #batch_input_params[idx, 0] = item['input_param']
        # batch_input_params[idx, 1] = item['input_type']

    return (batch_input_images,  batch_target_images)
    #return (batch_input_images, batch_input_params, batch_target_images)

def load_tiff_with_gamma(input_path, gamma=2.2):
    # 1. 读取图像
    input = tifffile.imread(input_path)
    input = np.clip(input, 1e-6, None)
    if input is None:
        raise ValueError(f"Failed to read image at {input_path}")

    # 2. 灰度图处理成 [H, W, 1]
    if input.ndim == 2:
        input = np.expand_dims(input, axis=2)

    # 3. 转为 float32（避免整数爆炸）
    input = input.astype(np.float32)
    input = np.nan_to_num(input, nan=1e-6, posinf=1.0, neginf=1e-6)
    input = np.clip(input, 1e-6, None)
    # 4. 归一化到 [0, 1]（可选，但常用）
    input = input / (input.max() + 1e-6)

    # 5. Gamma 压缩（真正需要的操作）
    #input = np.clip(input, 1e-6, 1.0)  # 避免 0^x = nan
    gamma_mapped = input ** (1.0 / gamma)

    return gamma_mapped

class ArbitraryImageFolder(data.Dataset):

    # def __init__(self, input_paths, label_paths, params, type, hws):
    #     self.input_paths = input_paths
    #     self.label_paths = label_paths
    #     self.params = params
    #     self.type= type
    #     self.hws = hws
    #     self.size = len(input_paths)
    def __init__(self, input_paths, label_paths, hws):
        self.input_paths = input_paths
        self.label_paths = label_paths
        self.hws = hws
        self.size = len(input_paths)

    def __getitem__(self, index):
        input_path, label_path = self.input_paths[index], self.label_paths[index]
        img_h, img_w = self.hws[index]

        # input = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        # label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        #input = tifffile.imread(input_path)
        input = load_tiff_with_log(input_path)
        #print(input.shape, input.min(), input.max(),input.mean())
        label = tifffile.imread(label_path)

        if np.isnan(input).any() or np.isinf(input).any():
            print(f"\033[91m[NaN/Inf Detected] input at index {index}, path: {input_path}\033[0m")
            print(f"shape: {input.shape}, min: {np.min(input)}, max: {np.max(input)}, mean: {np.mean(input)}")
    
        
        if input.ndim == 2:  # 灰度图，只有 [H, W]
            input = np.expand_dims(input, axis=2)
        if label.ndim == 2:
            label = np.expand_dims(label, axis=2)

        input = input.transpose((2, 0, 1))
        input = torch.from_numpy(input).unsqueeze(0).float()
        label = label.transpose((2, 0, 1))
        label = torch.from_numpy(label).unsqueeze(0).float()

        #return {'input_img': input, 'target_img': label, 'input_param': para, 'input_type': type, 'input_height': img_h, 'input_width': img_w}
        return {'input_img': input, 'target_img': label, 'input_height': img_h, 'input_width': img_w,'input_path': input_path}
    def __len__(self):
        return self.size
    

##### change the saved model filename for another operator#####
#####模型名称######
name = 'train_L0L1_0.3_512_edge_operator'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
epoches = 200
batchsize = 12
num_workers = 12

edgeCompute = EdgeComputation()
edgeCompute2 = EdgeComputation2()
#需要替换
filter_input_folder = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR512_L0L1_input_0.3V2'
filter_input_folder_test = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR512_L0L1_valinput_0.3V2'

train_input_paths = []
train_label_paths = []
train_params = []
train_type = []
train_hws = []

#打开训练文件
with open("vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR_l0l1_train512_listV2_1.txt","r") as f:
    for line in f:
        label_path = line.strip()
        label_imgname = os.path.splitext(os.path.basename(label_path))[0]
        name_parts = label_imgname.split('_')
        #img_h, img_w = int(name_parts[-2]), int(name_parts[-1])        
        img_h = 512
        img_w = 512
        #type = (label_path.split('/')[3].split('_')[1])
        type = 'qwls'
        if type == 'qwls':
            #param = float(name_parts[-2])
            #param = float('%s.%s'%(param[:-5],param[-5:]))   #由para在文件名中的位置决定
            param = 1.0
            train_type.append(float('0.1'))
            train_params.append(param)
            base_imgname = label_imgname.replace('_%s'%(name_parts[-1]), '')
            #自己指定input的路径
            train_input_paths.append(os.path.join(filter_input_folder, base_imgname+'.tiff'))
            train_label_paths.append(label_path)
        else:
            continue
        
        train_hws.append((img_h, img_w))

#测试代码
test_input_paths = []
test_label_paths = []
test_params = []
test_type = []
test_hws = []
#所需文件路径
with open("vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR_l0l1_test512_listV2_1.txt","r") as f:
    lines = sorted(f.readlines())
    for line in lines:
        label_path = line.strip()
        label_imgname = os.path.splitext(os.path.basename(label_path))[0]
        name_parts = label_imgname.split('_')
        #type = (label_path.split('/')[3].split('_')[1])
        #img_h, img_w = int(name_parts[-1]), int(name_parts[-2])
        img_h = 512
        img_w = 512
        type = 'qwls'
        if type == 'qwls':
            #param = float(name_parts[-3])
            para = 1.0
            
            #param = float('%s.%s'%(para[:-5],para[-5:]))
            test_type.append(float('0.1'))
            test_params.append(param)
            base_imgname = label_imgname.replace('_%s'%(name_parts[-1]), '')
            test_input_paths.append(os.path.join(filter_input_folder_test, base_imgname+'.tiff'))
            test_label_paths.append(label_path)
        else:
            continue
        
        test_hws.append((img_h, img_w))

def gradient(x):
    # x: [B,1,H,W]
    gx = x[:, :, :, :-1] - x[:, :, :, 1:]
    gy = x[:, :, :-1, :] - x[:, :, 1:, :]
    return gx, gy

trainSetSize, testSetSize = len(train_input_paths), len(test_input_paths)
epochSize = 279 ###需根据输入设定

model = SimpleCNN()
model = torch.nn.DataParallel(model.cuda())

criterion = torch.nn.MSELoss(size_average=True)
pixel_criterion = torch.nn.L1Loss(reduction='mean').cuda()
criterion = criterion.cuda()
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#train_dataset = ArbitraryImageFolder(train_input_paths, train_label_paths, train_params, train_type, train_hws)
#无参数输入训练
train_dataset = ArbitraryImageFolder(train_input_paths, train_label_paths, train_hws)
train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False, collate_fn=var_custom_collate,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

#test_dataset = ArbitraryImageFolder(test_input_paths, test_label_paths, test_params, test_type, test_hws)
test_dataset = ArbitraryImageFolder(test_input_paths, test_label_paths, test_hws)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, collate_fn=var_custom_collate,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

total_start_time = time.time()
iter = 0
iterTest = 0
######Tensorboard 写入日志路径########
writer = SummaryWriter('vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/checkpoints/l0l1_filter_512_edge_log')
for p in range(epoches):
    train_loss = 0
    test_loss = 0
    train_count = 0
    test_count = 0
    start_time = time.time()
    if p == 85:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    for curr_iter, batch_data in enumerate(train_dataloader):
        iter += 1

        inputTensor,labelTensor = batch_data
        inputTensor = inputTensor - 0.5
        # broadcast to (B, 3, H, W)
        #labelTensor = labelTensor / 255.0  # 转为 0~1 区间
        labelTensor = labelTensor - 0.5 
        inputTensor_v = Variable(inputTensor.cuda())
        #inputParaTensor_v = Variable(inputParaTensor.cuda())
        labelTensor_v = Variable(labelTensor.cuda())
        #edge = Variable(edge.cuda())
        #pred_v = model(inputParaTensor_v, inputTensor_v)
        pred_v = model(inputTensor_v)
        optimizer.zero_grad()
        # mse_loss = criterion(pred_v, labelTensor_v)
        # per_pixel_mse = (pred_v - labelTensor_v).pow(2)         # [B, C, H, W]
        # weight_map    = 1.0 + 2.0 * edge       # [B, 1, H, W]
        # weight_map    = weight_map.expand_as(pred_v)    # → [B, C, H, W]
        # weighted_mse  = (per_pixel_mse * weight_map).mean()
        # loss = mse_loss + 0.5 * weighted_mse
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

        inputTensor,labelTensor = batch_data
        inputTensor = inputTensor - 0.5
        #labelTensor = labelTensor / 255.0  # 转为 0~1 区间
        labelTensor = labelTensor - 0.5 
        inputTensor_v = Variable(inputTensor.cuda())
        #inputParaTensor_v = Variable(inputParaTensor.cuda())
        labelTensor_v = Variable(labelTensor.cuda())

        pred_v = model(inputTensor_v)
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
    #torch.save(model.module.state_dict(),'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/models/model_{}_{}_qwls_single.net'.format(name,p+1))
    if (p + 1) % 100 == 0 :
        torch.save(
            model.module.state_dict(),
            #######模型保存路径###########
            f'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/models/model_{name}_{p+1}_l0l1_filter_edge_512.net'
        )
total_elapsed = time.time() - total_start_time
writer.close()
print('Total time elapsed: %.2f days'%(total_elapsed/(3600*24)))
