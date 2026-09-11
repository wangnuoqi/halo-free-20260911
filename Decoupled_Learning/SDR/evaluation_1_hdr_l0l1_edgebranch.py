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
#train_operator_l0l1filter.py
MODEL_PATH = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/models/model_train_L0L1_0.3_512_edgebranch_operatorV2_200_l0l1_filter_edge_512.net'
#train_operator_l0l1filterV2.py
#MODEL_PATH = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/models/model_train_L0L1_512_operatorV3(2)_200_l0l1_filter_0.3.net'

TEST_LIST_PATH = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR_l0l1_test512_listV2_1.txt'
INPUT_FOLDER = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR512_L0L1_valinput_0.3V2'

SAVE_FOLDER = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRtest_output_l0l1_edgebranchV2'
#SAVE_FOLDER = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRtest_output_l0l1_edge'
def save_image_png(filename, data):
    img = data.numpy()
    img = np.clip(img,0,255)
    img = img.transpose(1, 2, 0).astype("uint8")

    cv2.imwrite(filename,img)

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

class DoubleBranchNetwork(nn.Module):
    def __init__(self):
        super(DoubleBranchNetwork, self).__init__()
        
        # 输入图像分支
        self.input_branch = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU()
        )
        
        # 边缘图像分支
        self.edge_branch = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU()
        )
        
        # 共享残差块
        self.resblocks = nn.Sequential(
            ResidualBlock(128, dilation=2),  # 输入通道数调整为128，因为两个分支合并
            ResidualBlock(128, dilation=4),
            ResidualBlock(128, dilation=8),
            ResidualBlock(128, dilation=1)
        )
        
        # 上采样和输出层
        #self.up = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.norm_up = nn.InstanceNorm2d(64, affine=True)
        
        self.conv_out1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.norm_out1 = nn.InstanceNorm2d(64, affine=True)
        self.conv_out2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
                # 应用Kaiming初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """递归初始化所有卷积层"""
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.InstanceNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x_input, x_edge):
        # 处理输入图像分支
        input_feat = self.input_branch(x_input)
        
        # 处理边缘图像分支
        edge_feat = self.edge_branch(x_edge)
        
        # 合并两个分支的特征
        merged_feat = torch.cat([input_feat, edge_feat], dim=1)
        
        # 通过残差块
        x = self.resblocks(merged_feat)
        
        # 上采样和输出
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
        y = torch.sum(y,1)/3
        y /= 4
        return y
    
class EdgeComputation2(nn.Module):
    def __init__(self):
        super(EdgeComputation2, self).__init__()

    def forward(self, path):
        # 3. 读取边缘图
        edge_img = tifffile.imread(path)  # shape: [H, W]
        if edge_img is None:
            raise FileNotFoundError(f"[Edge Missing] Cannot find edge file: {path}")
        
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
    batch_edges = torch.Tensor(len(batch), 1, min_h, min_w)
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
        batch_edges[idx] =  item['edge'][:, :, off_y:off_y+min_h, off_x:off_x+min_w]
        #batch_input_images[idx,1] = edgeCompute2(item['input_path'],
        #         edge_root='/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR512_l0l1_label_edgeV2_train_test',
        #         edge_suffix='_l1l0.tiff')
        #batch_target_images[idx] = item['target_img'][:, :, off_y:off_y+min_h, off_x:off_x+min_w]
        #batch_input_params[idx, 0] = item['input_param']
        # batch_input_params[idx, 1] = item['input_type']

    return (batch_input_images,  batch_target_images, batch_edges)

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
    def __init__(self, input_paths, label_paths, hws, edges_paths):
        self.input_paths = input_paths
        self.label_paths = label_paths
        self.edges_paths = edges_paths
        self.hws = hws
        self.size = len(input_paths)

    def __getitem__(self, index):
        input_path, label_path, edges_path = self.input_paths[index], self.label_paths[index],self.edges_paths[index]
        img_h, img_w = self.hws[index]
        edges_path = self.edges_paths[index]
        # input = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        # label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        #input = tifffile.imread(input_path)
        input = load_tiff_with_log(input_path)
        #print(input.shape, input.min(), input.max(),input.mean())
        label = tifffile.imread(label_path)
        edge = edgeCompute2(edges_path)
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
        edge = edge.unsqueeze(0).float()
        #return {'input_img': input, 'target_img': label, 'input_param': para, 'input_type': type, 'input_height': img_h, 'input_width': img_w}
        return {'input_img': input, 'target_img': label, 'input_height': img_h, 'input_width': img_w,'input_path': input_path, 'edge':edge}
    def __len__(self):
        return self.size


# 读取测试文件路径、参数等（根据你训练的方式）
test_input_paths = []
test_label_paths = []
test_edges_paths = []
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
            edge_root='/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR512_l0l1_label_edgeV2_train_test'
            test_type.append(float('0.1'))
            test_params.append(param)
            base_imgname = label_imgname.replace('_%s'%(name_parts[-1]), '')

            saveName = os.path.join(base_imgname + '_pred.tiff')
            test_input_paths.append(os.path.join(INPUT_FOLDER, base_imgname+'.tiff'))
            test_label_paths.append(label_path)
            test_edges_paths.append(os.path.join(edge_root, base_imgname+'_l1l0.tiff'))
            saveNameList.append(os.path.join(SAVE_FOLDER, saveName))
            test_hws.append((img_h, img_w))

# 构造 DataLoader
test_dataset = ArbitraryImageFolder(test_input_paths, test_label_paths, test_hws, test_edges_paths)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=var_custom_collate,
                              num_workers=0, pin_memory=True, drop_last=True)

# 加载模型
model = DoubleBranchNetwork()
model.load_state_dict(torch.load(MODEL_PATH))
model = model.cuda()
model.eval()

criterion = torch.nn.MSELoss()

total_loss = 0.0
count = 0

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        inputTensor, labelTensor, edgeTensor = batch
        inputTensor = inputTensor - 0.5
        labelTensor = labelTensor - 0.5
        edgeTensor = edgeTensor - 0.5

        # edge = edgeCompute( inputTensor)
        # for idx in range(edge.size(0)):  # 遍历 batch 内每张图
        #     y = edge[idx]
        #     edge_np = y.squeeze().detach().cpu().numpy()
        #     edge_np = (edge_np * 255.0).clip(0, 255).astype('uint8')                         # [1, H, W]
            
            #save_path = f"vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRtest_output/edge_{i}_{idx}.png"
            #cv2.imwrite(save_path, edge_np)
        
        inputTensor_v = Variable(inputTensor.cuda())
        labelTensor_v = Variable(labelTensor.cuda())
        edgeTensor_V = Variable(edgeTensor.cuda())

        pred = model(inputTensor_v,edgeTensor_V)
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
