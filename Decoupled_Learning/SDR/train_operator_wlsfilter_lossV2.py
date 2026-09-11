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
#
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
        # para = para.view(B, -1, 1, 1)      # [B, D, 1, 1]
        # para = para.expand(-1, -1, H, W) 
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

#图像边缘计算    
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
        y = torch.sum(y,1)/3.0
        y /= 4
        #y_np = y.numpy().astype(np.float32)
        # output_path = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/123.tiff"
        # tifffile.imwrite(output_path, y_np.astype(np.float32), photometric='minisblack')
        # print("Exists after write?", os.path.exists(output_path))
        return y


def var_custom_collate(batch):
    min_h, min_w = 10000, 10000
    for item in batch:
        min_h = min(min_h, item['input_height'])
        min_w = min(min_w, item['input_width'])

    batch_input_images = torch.Tensor(len(batch), 2, min_h, min_w)
    batch_target_images = torch.Tensor(len(batch), 1, min_h, min_w)
    #batch_input_params = torch.Tensor(len(batch), 1)
    for idx, item in enumerate(batch):
        off_y = random.randint(0, item['input_height']-min_h)
        off_x = random.randint(0, item['input_width']-min_w)
        batch_input_images[idx,0:1] = item['input_img'][:, :, off_y:off_y+min_h, off_x:off_x+min_w]
        batch_input_images[idx,1] = edgeCompute(batch_input_images.narrow(0,idx,1).narrow(1,0,1))
        batch_target_images[idx] = item['target_img'][:, :, off_y:off_y+min_h, off_x:off_x+min_w]
        #batch_input_params[idx, 0] = item['input_param']
        # batch_input_params[idx, 1] = item['input_type']

    return (batch_input_images,  batch_target_images)
    #return (batch_input_images, batch_input_params, batch_target_images)

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
    #img = img / (img.max() + 1e-6)
    log_img = np.log(img)
    min_val = log_img.min()
    max_val = 0.0  # 等于 log(1)
    log_norm = (log_img - min_val) / (max_val - min_val + 1e-6)
    #output_path = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/123.tiff"
    #tifffile.imwrite(output_path, log_norm.astype(np.float32), photometric='minisblack')
    #print("Exists after write?", os.path.exists(output_path))
    return log_norm

class ArbitraryImageFolder(data.Dataset):
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
            print(f"    shape: {input.shape}, min: {np.min(input)}, max: {np.max(input)}, mean: {np.mean(input)}")
    
        if np.isnan(label).any() or np.isinf(label).any():
            print(f"\033[91m[NaN/Inf Detected] label at index {index}, path: {label_path}\033[0m")
            print(f"    shape: {label.shape}, min: {np.min(label)}, max: {np.max(label)}, mean: {np.mean(label)}")

        if input.ndim == 2:  # 灰度图，只有 [H, W]
            input = np.expand_dims(input, axis=2)
        if label.ndim == 2:
            label = np.expand_dims(label, axis=2)

        input = input.transpose((2, 0, 1))
        input = torch.from_numpy(input).unsqueeze(0).float()
        label = label.transpose((2, 0, 1))
        label = torch.from_numpy(label).unsqueeze(0).float()

        #return {'input_img': input, 'target_img': label, 'input_param': para, 'input_type': type, 'input_height': img_h, 'input_width': img_w}
        return {'input_img': input, 'target_img': label, 'input_height': img_h, 'input_width': img_w}
    def __len__(self):
        return self.size
    

##### change the saved model filename for another operator##### 
name = 'train_wls_512_operator_l0lossV2(1)'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
epoches = 200
batchsize = 12
num_workers = 12

edgeCompute = EdgeComputation()
#需要替换
filter_input_folder = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR512_input'
filter_input_folder_test = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR512_valinput'

# filter_input_folder = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR1024_L0L1_input'
# filter_input_folder_test = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR1024_L0L1_valinput'

train_input_paths = []
train_label_paths = []
train_params = []
train_type = []
train_hws = []

#打开训练文件、
#####################通过 try2 构建的训练文件路径
with open("vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR_wls_train512_list.txt","r") as f:
    for line in f:
        label_path = line.strip()
        label_imgname = os.path.splitext(os.path.basename(label_path))[0]
        name_parts = label_imgname.split('_')
        #img_h, img_w = int(name_parts[-2]), int(name_parts[-1])
        ######更改输入图像大小###########        
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
with open("vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR_wls_test512_list.txt","r") as f:
    for line in f:
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

trainSetSize, testSetSize = len(train_input_paths), len(test_input_paths)
epochSize = 279 ###需根据输入设定

model = SimpleCNN()
mse_criterion   = torch.nn.MSELoss().cuda()
l1_criterion    = torch.nn.L1Loss().cuda()
criterion = torch.nn.MSELoss(size_average=True).cuda()
model = torch.nn.DataParallel(model.cuda())
criterion = criterion.cuda()
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
#train_dataset = ArbitraryImageFolder(train_input_paths, train_label_paths, train_params, train_type, train_hws)
#无参数输入训练
train_dataset = ArbitraryImageFolder(train_input_paths, train_label_paths, train_hws)
train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, collate_fn=var_custom_collate,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

#test_dataset = ArbitraryImageFolder(test_input_paths, test_label_paths, test_params, test_type, test_hws)
test_dataset = ArbitraryImageFolder(test_input_paths, test_label_paths, test_hws)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, collate_fn=var_custom_collate,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

total_start_time = time.time()
iter = 0
iterTest = 0
writer = SummaryWriter('vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/checkpoints/wls_log')
switch_epoch = 100

def get_edge(img):
    # 简单用Sobel或差分算子
    edge_x = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    edge_y = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    edge = torch.zeros_like(img)
    edge[:, :, :, 1:] += edge_x
    edge[:, :, :, :-1] += edge_x
    edge[:, :, 1:, :] += edge_y
    edge[:, :, :-1, :] += edge_y
    return edge / 4.0  # 归一化处理

class GradientConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        pred_edge = get_edge(pred)
        target_edge = get_edge(target)
        return F.l1_loss(pred_edge, target_edge)  # 或者MSELoss

for p in range(epoches):
    train_loss = 0
    test_loss = 0
    train_count = 0
    test_count = 0
    start_time = time.time()
    if p == 85:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if p == 100:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    if p == 180:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for curr_iter, batch_data in enumerate(train_dataloader):
        iter += 1

        inputTensor,labelTensor = batch_data
        inputTensor = inputTensor - 0.5
        #labelTensor = labelTensor / 255.0  # 转为 0~1 区间
        labelTensor = labelTensor - 0.5 
        inputTensor_v = Variable(inputTensor.cuda())
        #inputParaTensor_v = Variable(inputParaTensor.cuda())
        labelTensor_v = Variable(labelTensor.cuda())

        optimizer.zero_grad()
        #pred_v = model(inputParaTensor_v, inputTensor_v)
        pred_v = model(inputTensor_v)
        if p < switch_epoch:
            # 阶段 1：只用 MSE
            loss = mse_criterion(pred_v, labelTensor_v)
        else:
            # 阶段 2：梯度一致性
            pixel_loss = F.l1_loss(pred_v, labelTensor_v)
            edge_loss = GradientConsistencyLoss()(pred_v, labelTensor_v)
            mse_loss = mse_criterion(pred_v, labelTensor_v)
            loss = 0.5*edge_loss + pixel_loss
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
        writer.add_scalar('Loss/test_batch', loss.item(), iter)
        if iterTest % int(epochSize/(batchsize*20)) == 0:
            break

    train_loss = train_loss / train_count
    test_loss = test_loss / test_count
    elapsed = time.time() - start_time
    print('Epoch: %d average train loss: %.4f'%(p+1,train_loss))
    print('Epoch: %d average test  loss: %.4f'%(p+1,test_loss))
    print('Epoch: %d time elapsed: %.2f hours'%(p+1,elapsed/3600))
    #torch.save(model.module.state_dict(),'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/models/model_{}_{}_qwls_single.net'.format(name,p+1))
    if (p + 1) % 200 == 0 :
        torch.save(
            model.module.state_dict(),
            f'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/models/model_{name}.net'
        )
total_elapsed = time.time() - total_start_time
writer.close()
print('Total time elapsed: %.2f days'%(total_elapsed/(3600*24)))
