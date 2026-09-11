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

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation, norm_layer=nn.Identity, use_dropout=False, use_bias=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, dilation, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, dilation, norm_layer, use_dropout, use_bias):
        conv_block = []

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, bias=use_bias),
                       PALayer(dim), CALayer(dim),
                       norm_layer(dim),
                       nn.PReLU()]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, bias=use_bias),
                       PALayer(dim), CALayer(dim),
                       norm_layer(dim),
                       nn.PReLU()]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class PALayer(nn.Module):
    """像素级注意力模块"""
    def __init__(self, channel: int):
        super(PALayer, self).__init__()
        self.pa0 = nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.pa1 = nn.Sequential(
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pa1(self.relu(self.pa0(x)))
        return x * y

class CALayer(nn.Module):
    """通道级注意力模块"""
    def __init__(self, channel: int):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca0 = nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True)
        self.relu0 = nn.ReLU(inplace=True)
        self.ca1 = nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.sig(self.ca1(self.relu0(self.ca0(y))))
        return x * y

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=None, stride=1,
                 norm=False, bias=True, dilation=1):
        super(DownBlock, self).__init__()
        padding = padding or dilation * (kernel_size - 1) // 2

        self.model = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, bias=bias, dilation=dilation),
            PALayer(out_channels),
            CALayer(out_channels)
        )
        self.norm = nn.InstanceNorm2d(out_channels) if norm else None
        self.relu = nn.PReLU()
    
    def forward(self, input_features):
        input_features = self.model(input_features)
        if self.norm:
            input_features = self.norm(input_features)
        out_features = self.relu(input_features)
        return out_features

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=None, stride=1,
                 norm=False, bias=True, dilation=1):
        super(UpBlock, self).__init__()
        padding = padding or dilation * (kernel_size - 1) // 2

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, bias=bias, dilation=dilation),
            PALayer(out_channels),
            CALayer(out_channels)
        )
        self.norm = nn.InstanceNorm2d(out_channels) if norm else None
        self.relu = nn.PReLU()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.model(x)
        if self.norm:
            x = self.norm(x)
        out_features = self.relu(x)
        return out_features

class UNet(nn.Module):
    def __init__(self, in_channels=32):
        super(UNet, self).__init__()
        self.down1 = DownBlock(in_channels, in_channels)
        self.down2 = DownBlock(in_channels, in_channels)
        self.down3 = DownBlock(in_channels, in_channels)
        self.down4 = DownBlock(in_channels, in_channels)
        self.up1 = UpBlock(in_channels, in_channels)
        self.up2 = UpBlock(in_channels, in_channels)
        self.up3 = UpBlock(in_channels, in_channels)
        self.up4 = UpBlock(in_channels, in_channels)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.up4(u3, x)
        return u4

class CustomNetwork(nn.Module):
    def __init__(self, input_channels=2, output_channels=1, features=32):
        """
        定制化网络模型，适配2维输入和1维输出
        input_channels: 输入通道数，默认为2
        output_channels: 输出通道数，默认为1
        features: 基础特征数
        """
        super(CustomNetwork, self).__init__()
        
        # 输入处理层 - 将2维输入映射到特征空间
        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, features, kernel_size=3, stride=1, padding=1),
            PALayer(features),
            CALayer(features),
            nn.PReLU()
        )
        
        # 主干网络 - 保留原有的UNet结构
        self.backbone0 = nn.Sequential(
            ResnetBlock(features, 1),
            ResnetBlock(features, 1)
        )
        
        self.backbone1 = nn.Sequential(
            ResnetBlock(features, 1),
            ResnetBlock(features, 1),
            nn.MaxPool2d(2)
        )
        
        self.backbone2 = nn.Sequential(
            ResnetBlock(features, 1),
            ResnetBlock(features, 1),
            nn.MaxPool2d(2)
        )
        
        # 上采样和特征融合
        self.up1 = UpBlock(features, features)
        self.up21 = UpBlock(features, features)
        self.up22 = UpBlock(features, features)
        
        # UNet主体
        self.body0 = UNet(features)
        self.body1 = UNet(features)
        self.body2 = UNet(features)
        
        # 输出层 - 将特征映射到1维输出
        self.output_layer = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            PALayer(features),
            CALayer(features),
            nn.PReLU(),
            nn.Conv2d(features, output_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        """
        前向传播函数
        x: 输入张量，形状为(batch_size, 2, height, width)
        """
        # 输入投影
        x = self.input_projection(x)
        
        # 主干网络特征提取
        body_stage0 = self.body0(x)
        in_stage1 = self.backbone1(body_stage0)
        body_stage1 = self.body1(in_stage1)
        in_stage2 = self.backbone2(body_stage1)
        body_stage2 = self.body2(in_stage2)
        
        # 上采样和特征融合
        out_stage2 = self.output_layer(self.up21(self.up22(body_stage2, body_stage1), body_stage0))
        
        return out_stage2

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
        # y_np = y.numpy().astype(np.float32)
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
            print(f"    shape: {input.shape}, min: {np.min(input)}, max: {np.max(input)}, mean: {np.mean(input)}")
    
        
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
name = 'train_L0L1_512_operator_unet'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
epoches = 200
batchsize = 12
num_workers = 12

edgeCompute = EdgeComputation()
#需要替换
filter_input_folder = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR512_L0L1_input_0.3V2'
filter_input_folder_test = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR512_L0L1_valinput_0.3V2'

# filter_input_folder = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR1024_L0L1_input'
# filter_input_folder_test = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR1024_L0L1_valinput'

train_input_paths = []
train_label_paths = []
train_params = []
train_type = []
train_hws = []

#打开训练文件、
#####################通过 try2 构建的训练文件路径
with open("vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR_l0l1_train512_listV2_1.txt","r") as f:
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
with open("vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR_l0l1_test512_listV2_1.txt","r") as f:
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

model = CustomNetwork(input_channels=2, output_channels=1)
criterion = torch.nn.MSELoss(size_average=True)
model = torch.nn.DataParallel(model.cuda())
criterion = criterion.cuda()
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

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
writer = SummaryWriter('vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/checkpoints/l0l1_filter_log')
for p in range(epoches):
    train_loss = 0
    test_loss = 0
    train_count = 0
    test_count = 0
    start_time = time.time()
    if p == 85:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

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
        loss = criterion(pred_v, labelTensor_v)
        #loss_hist = histogram_loss(pred_v, labelTensor_v)
        #print(f"[Debug] inputTensor: min={inputTensor.min().item():.4f}, max={inputTensor.max().item():.4f}")
        #print(f"[Debug] labelTensor: min={labelTensor.min().item():.4f}, max={labelTensor.max().item():.4f}")
        #print(f"[Debug] pred_v: min={pred_v.min().item():.4f}, max={pred_v.max().item():.4f}")
        #loss = 0.5*loss+0.5*loss_hist
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
            f'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/models/model_{name}_{p+1}_l0l1_filter_0.3.net'
        )
total_elapsed = time.time() - total_start_time
writer.close()
print('Total time elapsed: %.2f days'%(total_elapsed/(3600*24)))
