import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
#from model import SimpleCNN, ArbitraryImageFolder, var_custom_collate  # 确保你模块分离或者这些类和函数能 import
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile

# 设置模型路径和数据路径
MODEL_PATH = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/models/model_train_L0L1_512_operator_unet_200_l0l1_filter_0.3.net'

TEST_LIST_PATH = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR_l0l1_test512_listV2_1.txt'
INPUT_FOLDER = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR512_L0L1_valinput_0.3V2'

SAVE_FOLDER = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRtest_output_l0l1_unet'

def save_image_tiff(filename, data):
    # 如果是 torch.Tensor，先转 numpy
    if hasattr(data, 'cpu'):
        arr = data.cpu().numpy()
        arr = arr/255
        arr = np.clip(arr,0,1)
    else:
        arr = data
    # 转成 HWC
    arr = np.transpose(arr, (1, 2, 0))
    # 确保 float32
    arr = arr.astype(np.float32)
    # 写入
    tifffile.imwrite(filename, arr)

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
        y = torch.sum(y,1)/3
        y /= 4
        return y
edgeCompute = EdgeComputation()

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

# 读取测试文件路径、参数等（根据你训练的方式）
test_input_paths = []
test_label_paths = []
test_params = []
test_type = []
test_hws = []
saveNameList = []
with open(TEST_LIST_PATH, "r") as f:
    for line in f:
        label_path = line.strip()
        label_imgname = os.path.splitext(os.path.basename(label_path))[0]
        name_parts = label_imgname.split('_')

        img_h = 512
        img_w = 512
        type = 'qwls'

        if type == 'qwls':
            #inputPath = "vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR_fullpic_valinput"
            #searchPath = "vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR_fullpic_valinput/*.tiff"

            param = 1.0
            test_type.append(float('0.1'))
            test_params.append(param)
            base_imgname = label_imgname.replace('_%s'%(name_parts[-1]), '')
            saveName = os.path.join(base_imgname + '_pred.tiff')
            test_input_paths.append(os.path.join(INPUT_FOLDER, base_imgname+'.tiff'))
            test_label_paths.append(label_path)
            saveNameList.append(os.path.join(SAVE_FOLDER, saveName))
            test_hws.append((img_h, img_w))

# 构造 DataLoader
test_dataset = ArbitraryImageFolder(test_input_paths, test_label_paths, test_hws)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=var_custom_collate,
                              num_workers=0, pin_memory=True, drop_last=True)

# 加载模型
model = CustomNetwork()
model.load_state_dict(torch.load(MODEL_PATH))
model = model.cuda()
model.eval()

criterion = torch.nn.MSELoss()

total_loss = 0.0
count = 0

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        inputTensor, labelTensor = batch
        inputTensor = inputTensor - 0.5
        #labelTensor = labelTensor / 255.0  # 转为 0~1 区间
        labelTensor = labelTensor - 0.5

        #  
        # edge = edgeCompute( inputTensor)
        # for idx in range(edge.size(0)):  # 遍历 batch 内每张图
        #     y = edge[idx]
        #     edge_np = y.squeeze().detach().cpu().numpy()
        #     edge_np = (edge_np * 255.0).clip(0, 255).astype('uint8')                         # [1, H, W]
            
            #save_path = f"vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRtest_output/edge_{i}_{idx}.png"
            #cv2.imwrite(save_path, edge_np)
        
        inputTensor_v = Variable(inputTensor.cuda())
        #inputParaTensor_v = Variable(inputParaTensor.cuda())
        labelTensor_v = Variable(labelTensor.cuda())

        pred = model(inputTensor_v)
        loss = criterion(pred, labelTensor_v)

        total_loss += loss.item()
        count += 1
        pred = pred.data[0].cpu()
        output = (pred+0.5)*255
        save_path = saveNameList[i]
        save_image_tiff(save_path,output)
        print(f"[{i}] Loss: {loss.item():.6f}")

avg_loss = total_loss / count
print(f"\n Average test loss: {avg_loss:.6f}")
