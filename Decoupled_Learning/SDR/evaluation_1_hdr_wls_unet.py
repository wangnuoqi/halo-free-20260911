import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
#from model import SimpleCNN, ArbitraryImageFolder, var_custom_collate  # 确保你模块分离或者这些类和函数能 import
import torch.utils.data as data
import torch.nn as nn
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile

# 设置模型路径和数据路径
MODEL_PATH = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/models/model_train_wls_512_operator_unet.net'
TEST_LIST_PATH = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR_wls_test512_list.txt'
INPUT_FOLDER = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR512_valinput'
SAVE_FOLDER = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/WLS_test/HDRtest_output_wls_unet'

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

class SimpleAttnUNet(nn.Module):
    def __init__(self, input_channels=2, features=32, output_channels=1):
        super().__init__()
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, features, 3, padding=1),
            PALayer(features),
            CALayer(features),
            nn.PReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(features, features, 3, stride=2, padding=1),
            PALayer(features),
            CALayer(features),
            nn.PReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(features, features, 3, stride=2, padding=1),
            PALayer(features),
            CALayer(features),
            nn.PReLU()
        )
        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            PALayer(features),
            CALayer(features),
            nn.PReLU()
        )
        # 解码器
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features * 2, features, 3, padding=1),
            PALayer(features),
            CALayer(features),
            nn.PReLU()
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features * 2, features, 3, padding=1),
            PALayer(features),
            CALayer(features),
            nn.PReLU()
        )
        self.outconv = nn.Conv2d(features, output_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)                        # [B, features, H, W]
        e2 = self.enc2(e1)                       # [B, features, H/2, W/2]
        e3 = self.enc3(e2)                       # [B, features, H/4, W/4]
        b = self.bottleneck(e3)                  # [B, features, H/4, W/4]
        d2 = self.up2(torch.cat([b, e3], dim=1)) # [B, features, H/2, W/2]
        d1 = self.up1(torch.cat([d2, e2], dim=1))# [B, features, H, W]
        out = self.outconv(d1)                   # [B, 1, H, W]
        return out


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

edgeCompute = EdgeComputation()
edgeCompute2 = EdgeComputation2()

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
        batch_input_images[idx,1] = edgeCompute(batch_input_images.narrow(0,idx,1).narrow(1,0,1))

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
        return {'input_img': input, 'target_img': label, 'input_height': img_h, 'input_width': img_w,'input_path': input_path}
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
model = SimpleAttnUNet()
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
