import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

import os

from tqdm import tqdm
from guided_filter_pytorch.guided_filter import FastGuidedFilter
from guided_filter_pytorch.guided_filter import GuidedFilter
import cv2
import numpy as np
import time
import math

class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()
        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))
        self.bn  = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)

def weights_init_identity(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n_out, n_in, h, w = m.weight.data.size()
        if n_out < n_in:
            nn.init.xavier_uniform_(m.weight.data)
            return
        m.weight.data.zero_()
        ch, cw = h // 2, w // 2
        for i in range(n_in):
            m.weight.data[i, i, ch, cw] = 1.0
    elif classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)

def build_lr_net(norm=AdaptiveNorm, layer=5):
    layers = [
        nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
    ]

    for l in range(1, layer):
        layers += [nn.Conv2d(24,  24, kernel_size=3, stride=1, padding=2**l,  dilation=2**l,  bias=False),
                   norm(24),
                   nn.LeakyReLU(0.2, inplace=True)]

    layers += [
        nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(24,  1, kernel_size=1, stride=1, padding=0, dilation=1)
    ]

    net = nn.Sequential(*layers)

    net.apply(weights_init_identity)

    return net

class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8):
        super(DeepGuidedFilter, self).__init__()
        self.lr = build_lr_net()
        self.gf = FastGuidedFilter(radius, eps)

    def forward(self, x_lr, x_hr):
        return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)

    def init_lr(self, path):
        self.lr.load_state_dict(torch.load(path))

class GuidedFilterWithLearnableMap(DeepGuidedFilter):
    def __init__(self, radius=1, eps=1e-8):
        super(GuidedFilterWithLearnableMap, self).__init__()
        self.gf = GuidedFilter(radius, eps)

        # Learnable guided map module, 对输入的边缘做自适应调整，
        self.guided_map = nn.Sequential(
            nn.Conv2d(1, 64, 1, bias=False),
            AdaptiveNorm(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 1)
        )
        self.guided_map.apply(weights_init_identity)

########## -- 这里的guide_hr是引导图，即我们获得的edge map； 这里的 map img 是我们的coarsing filter img 用于在后续进行进一步的滤波 - #### 
########## -- 遵循v9中的操作，没有对我们的下采样网络进行训练，直接将我们的 target 输入进去后结果不是很好
##########  这里按照原文的操作，对输入做一个下采样后再输入到guided filter 中去。看看效果如何    
    def forward(self, x_hr, map_img):
        #guide_lr = self.guided_map(x_lr)
        guide_hr = self.guided_map(x_hr)
        return self.gf(guide_hr, map_img)

#### - 需要输入网络的基础信息图，用于推理得到真实的滤波图 - ####
##### 训练输入 #####
downsample_folder_train = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/down_sample_train" ## yes
input_folder_train = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRinput'
#guided_input_folder_train = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/guided_edge_train" ##yes


label_train = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRinput_label" ## yes

####### - 测试输入 - #########
coarsefiltered_folder_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/coarse_filtering_test'
guided_input_folder_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/guided_edge_test'

label_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/WLS-021544_V4'

##### - 数据加载 - #######
class GuidedFilterDataset(Dataset):
    def __init__(self, x_lr_dir, x_hr_dir, label_dir):
        self.x_lr_files = sorted([os.path.join(x_lr_dir, f) for f in os.listdir(x_lr_dir)])
        self.x_hr_files = sorted([os.path.join(x_hr_dir, f) for f in os.listdir(x_hr_dir)])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        print(f"LR: {len(self.x_lr_files)}, HR: {len(self.x_hr_files)}, Label: {len(self.label_files)}")

    def __len__(self):
        return min(len(self.x_lr_files), len(self.x_hr_files), len(self.label_files))

    def __getitem__(self, idx):
        x_lr   = cv2.imread(self.x_lr_files[idx], cv2.IMREAD_GRAYSCALE).astype('float32')
        x_hr   = cv2.imread(self.x_hr_files[idx], cv2.IMREAD_GRAYSCALE).astype('float32')
        label  = cv2.imread(self.label_files[idx], cv2.IMREAD_GRAYSCALE).astype('float32')
        input_hr_h, input_hr_w = x_hr.shape
        input_lr_h, input_lr_w = x_lr.shape

        return {
            'input_lr': torch.from_numpy(x_lr).unsqueeze(0).unsqueeze(0),  # [1,1,H,W]
            'input_hr': torch.from_numpy(x_hr).unsqueeze(0).unsqueeze(0),
            'target_img': torch.from_numpy(label).unsqueeze(0).unsqueeze(0),
            'input_height_hr': input_hr_h,
            'input_width_hr': input_hr_w,
            'input_height_lr': input_lr_h,
            'input_width_lr': input_lr_w
        }

def var_custom_collate(batch):
    min_h_hr = min([item['input_height_hr'] for item in batch])
    min_w_hr = min([item['input_width_hr'] for item in batch])
    min_h_lr = min([item['input_height_lr'] for item in batch])
    min_w_lr = min([item['input_width_lr'] for item in batch])   

    batch_x_hr = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_x_lr   = torch.zeros(len(batch), 1, min_h_lr, min_w_lr)
    batch_target_images = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)

    for idx, item in enumerate(batch):
        # 直接裁切到 min_h, min_w
        x_lr_crop = item['input_lr'][:, :, :min_h_lr, :min_w_lr]
        x_hr_crop   = item['input_hr'][:, :, :min_h_hr, :min_w_hr]
        label_crop  = item['target_img'][:, :, :min_h_hr, :min_w_hr]

        # 放到 batch
        batch_x_lr[idx] = x_lr_crop
        batch_x_hr[idx] = x_hr_crop
        batch_target_images[idx] = label_crop

    return batch_x_lr, batch_x_hr, batch_target_images


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device('cuda')
model =  DeepGuidedFilter(radius=1, eps=1e-8)
model = torch.nn.DataParallel(model).cuda()

train_dataset = GuidedFilterDataset(
    x_lr_dir = downsample_folder_train,
    x_hr_dir = input_folder_train,
    label_dir = label_train
)
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=12,
                          collate_fn=var_custom_collate, pin_memory=True, drop_last=True)

criterion = torch.nn.MSELoss(reduction='mean')

writer = SummaryWriter(log_dir="/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/log_dir")

######### psnr 计算 ###########
def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1.0 / mse)
    return psnr, mse

# def get_optimizer_and_scheduler(model, lr_1=0.01, lr_2=0.001, switch_epoch=0, total_epochs=150):
#     optimizer = optim.Adam(model.parameters(), lr=lr_1)

#     def lr_lambda(epoch):
#         # 前 switch_epoch：返回 1.0（保持 lr=lr_1）
#         # 后面：返回 lr_2 / lr_1（缩小到 0.01）
#         return 1.0 if epoch < switch_epoch else lr_2 / lr_1

#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
#     return optimizer, scheduler

save_debug_dir = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7"
os.makedirs(save_debug_dir, exist_ok=True)

def save_debug_images(batch_x_lr, batch_x_hr, output, batch_label, batch_idx):
    # 确保是 CPU 和 [0,1] 范围
    def norm_img(tensor):
        t = tensor.detach().cpu()
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)  # 归一化到0-1
        return t

    vutils.save_image(norm_img(batch_x_lr), f"{save_debug_dir}/batch_{batch_idx}_ilr.png")
    vutils.save_image(norm_img(batch_x_hr), f"{save_debug_dir}/batch_{batch_idx}_ihr.png")
    vutils.save_image(norm_img(output), f"{save_debug_dir}/batch_{batch_idx}_output.png")
    vutils.save_image(norm_img(batch_label), f"{save_debug_dir}/batch_{batch_idx}_label.png")

# 初始化
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 150

if __name__ == "__main__":
    for epoch in tqdm(range(epochs), desc="Training Progress", dynamic_ncols=True):
        model.train()
        epoch_mse = 0
        epoch_psnr = 0
        for batch_idx, (batch_x_lr, batch_x_hr, batch_label) in enumerate(train_loader):        
            t0 = time.time()

            batch_x_lr, batch_x_hr, batch_label = batch_x_lr.to(device), batch_x_hr.to(device), batch_label.to(device)
            #print(batch_edge.device, batch_coarse.device, batch_label.device)
            batch_x_lr = batch_x_lr - 128
            batch_x_hr = batch_x_hr - 128
            batch_label = batch_label/255
            
            #t1 = time.time()
            # 前向
            output = model(batch_x_lr, batch_x_hr)
            # if batch_idx % 10 == 0:
            #     save_debug_images( batch_x_lr, batch_x_hr, output, batch_label, batch_idx)
            #     print('已经保存了，狗熊')
            
            loss = criterion(output, batch_label)
            #t2 = time.time()

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #t3 = time.time()

            # 计算 PSNR 和 MSE
            if not torch.isfinite(output).all() or not torch.isfinite(batch_label).all():
                print("Warning: NaN or Inf detected in output or labels!")
                print("Output stats:", output.min().item(), output.max().item(), output.mean().item())
                print("Label stats:", batch_label.min().item(), batch_label.max().item(), batch_label.mean().item())

                # 打印前几个值（方便观察是否过大）
                print("Sample output values:", output.view(-1)[:10])
                print("Sample label values:", batch_label.view(-1)[:10])

            psnr, mse = calc_psnr(output.detach(), batch_label)
            if batch_idx % 200 == 0:
                print(f"iter MSE loss: {mse:.6f}")
            epoch_mse += mse
            epoch_psnr += psnr
            # 每 50 step 记录到 TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 50 == 0:
                writer.add_scalar('train/mse', mse, global_step)
                writer.add_scalar('train/psnr', psnr, global_step)
            #t4 = time.time()
            #if batch_idx % 20 == 0:
            #    print(f"[Batch {batch_idx}] Data: {t1-t0:.3f}s | Forward: {t2-t1:.3f}s | Backward: {t3-t2:.3f}s | Log: {t4-t3:.3f}s", flush=True)
        #scheduler.step()
            # 打印本 epoch 的平均指标
        print(f"Epoch {epoch+1}/{epochs} "
            f"- MSE: {epoch_mse/len(train_loader):.6f} "
            f"- PSNR: {epoch_psnr/len(train_loader):.2f} dB ")

        
    torch.save(model.state_dict(), "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/models/guided_filter_v14.pth")
    writer.close()


