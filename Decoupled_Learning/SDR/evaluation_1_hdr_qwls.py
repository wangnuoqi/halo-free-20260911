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


# 设置模型路径和数据路径
MODEL_PATH = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/models/model_train_L0L1_operator_200_l0l1_filter.net'
TEST_LIST_PATH = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR_l0l1_test512_list.txt'
INPUT_FOLDER = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR512_L0L1_valinput'
SAVE_FOLDER = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRtest_output_l0l1'
def save_image(filename, data):
    img = data.numpy()
    img = np.clip(img,0,255)
    img = img.transpose(1, 2, 0).astype("uint8")

    cv2.imwrite(filename,img)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(5, 64, kernel_size=3, padding=1)
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
        self.conv_out2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self,para, x_input):
        B, _, H, W = x_input.shape
        para = para.view(B, -1, 1, 1)      # [B, D, 1, 1]
        para = para.expand(-1, -1, H, W) 
        x = torch.cat([x_input, para], dim=1)
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
        y = torch.sum(y,1)/3
        y /= 4
        return y
edgeCompute = EdgeComputation()

def var_custom_collate(batch):
    min_h, min_w = 10000, 10000
    for item in batch:
        min_h = min(min_h, item['input_height'])
        min_w = min(min_w, item['input_width'])

    batch_input_images = torch.Tensor(len(batch), 4, min_h, min_w)
    batch_target_images = torch.Tensor(len(batch), 3, min_h, min_w)
    batch_input_params = torch.Tensor(len(batch), 1)
    for idx, item in enumerate(batch):
        off_y = random.randint(0, item['input_height']-min_h)
        off_x = random.randint(0, item['input_width']-min_w)
        batch_input_images[idx,0:3] = item['input_img'][:, :, off_y:off_y+min_h, off_x:off_x+min_w]
        batch_input_images[idx,3] = edgeCompute(batch_input_images.narrow(0,idx,1).narrow(1,0,3))
        #边缘保存
        # y = edgeCompute(batch_input_images.narrow(0,idx,1).narrow(1,0,3))
        # edge_np = y.squeeze().detach().cpu().numpy()
        # edge_np = (edge_np * 255.0).clip(0, 255).astype('uint8')
        # save_path = f'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRtest_output/edge_{idx}.png'
        #cv2.imwrite(save_path, edge_np)

        batch_target_images[idx] = item['target_img'][:, :, off_y:off_y+min_h, off_x:off_x+min_w]
        batch_input_params[idx, 0] = item['input_param']
        # batch_input_params[idx, 1] = item['input_type']

    return (batch_input_images, batch_input_params, batch_target_images)


class ArbitraryImageFolder(data.Dataset):

    def __init__(self, input_paths, label_paths, params, type, hws):
        self.input_paths = input_paths
        self.label_paths = label_paths
        self.params = params
        self.type= type
        self.hws = hws
        self.size = len(input_paths)

    def __getitem__(self, index):
        input_path, label_path, para, type = self.input_paths[index], self.label_paths[index], self.params[index], self.type[index]
        img_h, img_w = self.hws[index]

        input = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        # # 目标尺寸
        # target_size = (512, 512)

        # # Resize（保持数据类型）
        # input = cv2.resize(input, target_size, interpolation=cv2.INTER_LINEAR)
        # label = cv2.resize(label, target_size, interpolation=cv2.INTER_LINEAR)

        input = input.transpose((2, 0, 1))
        input = torch.from_numpy(input).unsqueeze(0).float()
        label = label.transpose((2, 0, 1))
        label = torch.from_numpy(label).unsqueeze(0).float()

        return {'input_img': input, 'target_img': label, 'input_param': para, 'input_type': type, 'input_height': img_h, 'input_width': img_w}

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

        img_h = 3002
        img_w = 4002
        type = 'qwls'

        if type == 'qwls':
            #inputPath = "vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR_fullpic_valinput"
            #searchPath = "vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR_fullpic_valinput/*.tiff"

            param = 1.0
            test_type.append(float('0.1'))
            test_params.append(param)
            base_imgname = label_imgname.replace('_%s_%s'%(name_parts[-2], name_parts[-1]), '')
            saveName = os.path.join(base_imgname + '_pred.png')
            test_input_paths.append(os.path.join(INPUT_FOLDER, base_imgname+'.tiff'))
            test_label_paths.append(label_path)
            saveNameList.append(os.path.join(SAVE_FOLDER, saveName))
            test_hws.append((img_h, img_w))

# 构造 DataLoader
test_dataset = ArbitraryImageFolder(test_input_paths, test_label_paths, test_params, test_type, test_hws)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=var_custom_collate, num_workers=0)

# 加载模型
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH))
model = model.cuda()
model.eval()

criterion = torch.nn.MSELoss()

total_loss = 0.0
count = 0

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        inputTensor, inputParaTensor, labelTensor = batch
        inputTensor = inputTensor - 0.5
        labelTensor = labelTensor / 255.0  # 转为 0~1 区间
        labelTensor = labelTensor - 0.5

        #  
        edge = edgeCompute( inputTensor)
        for idx in range(edge.size(0)):  # 遍历 batch 内每张图
            y = edge[idx]
            edge_np = y.squeeze().detach().cpu().numpy()
            edge_np = (edge_np * 255.0).clip(0, 255).astype('uint8')                         # [1, H, W]
            
            #save_path = f"vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRtest_output/edge_{i}_{idx}.png"
            #cv2.imwrite(save_path, edge_np)
        
        inputTensor_v = Variable(inputTensor.cuda())
        inputParaTensor_v = Variable(inputParaTensor.cuda())
        labelTensor_v = Variable(labelTensor.cuda())

        pred = model(inputParaTensor_v, inputTensor_v)
        loss = criterion(pred, labelTensor_v)

        total_loss += loss.item()
        count += 1
        pred = pred.data[0].cpu()
        output = (pred+0.5)*255
        save_path = saveNameList[i]
        save_image(save_path,output)
        print(f"[{i}] Loss: {loss.item():.6f}")

avg_loss = total_loss / count
print(f"\n Average test loss: {avg_loss:.6f}")
