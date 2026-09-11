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

class GuidedFilterWithLearnableMap(nn.Module):
    def __init__(self, radius=1, eps=1e-8):
        super(GuidedFilterWithLearnableMap, self).__init__()
        self.gf = GuidedFilter(radius, eps)

        # Learnable guided map module, 对输入的边缘做自适应调整，
        self.guided_map = nn.Sequential(
            nn.Conv2d(1, 15, 1, bias=False),
            AdaptiveNorm(15),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(15, 1, 1)
        )
        self.guided_map.apply(weights_init_identity)
########## -- 这里的guide_hr是引导图，即我们获得的edge map； 这里的 map img 是我们的coarsing filter img 用于在后续进行进一步的滤波 - #### 
    def forward(self, x_hr, map_img):
        #guide_lr = self.guided_map(x_lr)
        guide_hr = self.guided_map(x_hr)
        return self.gf(guide_hr, map_img)

#### - 需要输入网络的基础信息图，用于推理得到真实的滤波图 - ####
##### 训练输入 #####
coarsefiltered_folder_train = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/coarse_filtering_train" ## yes
guided_input_folder_train = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/guided_edge_train" ##yes

label_train = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRinput_label" ## yes

####### - 测试输入 - #########
coarsefiltered_folder_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/coarse_filtering_test'
guided_input_folder_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/guided_edge_test'

label_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/WLS-021544_V4'

##### - 数据加载 - #######
class GuidedFilterDataset(Dataset):
    def __init__(self, coarse_dir, edge_dir, label_dir):
        self.coarse_files = sorted([os.path.join(coarse_dir, f) for f in os.listdir(coarse_dir)])
        self.edge_files   = sorted([os.path.join(edge_dir, f) for f in os.listdir(edge_dir)])
        self.label_files  = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])

    def __len__(self):
        return len(self.coarse_files)

    def __getitem__(self, idx):
        coarse = cv2.imread(self.coarse_files[idx], cv2.IMREAD_GRAYSCALE).astype('float32')
        edge   = cv2.imread(self.edge_files[idx], cv2.IMREAD_GRAYSCALE).astype('float32')
        label  = cv2.imread(self.label_files[idx], cv2.IMREAD_GRAYSCALE).astype('float32')
        input_h, input_w = coarse.shape

        return {
            'input_img': torch.from_numpy(coarse).unsqueeze(0).unsqueeze(0),  # [1,1,H,W]
            'input_edge': torch.from_numpy(edge).unsqueeze(0).unsqueeze(0),
            'target_img': torch.from_numpy(label).unsqueeze(0).unsqueeze(0),
            'input_height': input_h,
            'input_width': input_w
        }

def var_custom_collate(batch):
    min_h = min([item['input_height'] for item in batch])
    min_w = min([item['input_width'] for item in batch])

    batch_coarse_images = torch.zeros(len(batch), 1, min_h, min_w)
    batch_edge   = torch.zeros(len(batch), 1, min_h, min_w)
    batch_target_images = torch.zeros(len(batch), 1, min_h, min_w)

    for idx, item in enumerate(batch):
        # 直接裁切到 min_h, min_w
        coarse_crop = item['input_img'][:, :, :min_h, :min_w]
        edge_crop   = item['input_edge'][:, :, :min_h, :min_w]
        label_crop  = item['target_img'][:, :, :min_h, :min_w]

        # 放到 batch
        batch_coarse_images[idx] = coarse_crop
        batch_edge[idx] = edge_crop
        batch_target_images[idx] = label_crop

    return batch_coarse_images, batch_edge, batch_target_images


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = torch.device('cuda')
model = GuidedFilterWithLearnableMap(radius=2, eps=1e-4)
model = torch.nn.DataParallel(model).cuda()

train_dataset = GuidedFilterDataset(
    coarse_dir = coarsefiltered_folder_train,
    edge_dir = guided_input_folder_train,
    label_dir = label_train
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0,
                          collate_fn=var_custom_collate, pin_memory=True, drop_last=True)

criterion = torch.nn.MSELoss(reduction='mean')

writer = SummaryWriter(log_dir="/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/log_dir")

def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(255.0* 255.0 / mse)
    return psnr, mse

def get_optimizer_and_scheduler(model, lr_1=0.01, lr_2=0.001, switch_epoch=40, total_epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=lr_1)

    def lr_lambda(epoch):
        # 前 switch_epoch：返回 1.0（保持 lr=lr_1）
        # 后面：返回 lr_2 / lr_1（缩小到 0.01）
        return 1.0 if epoch < switch_epoch else lr_2 / lr_1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler

save_debug_dir = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7"
os.makedirs(save_debug_dir, exist_ok=True)

def save_debug_images(batch_edge, batch_coarse, output, batch_label, batch_idx):
    # 确保是 CPU 和 [0,1] 范围
    def norm_img(tensor):
        t = tensor.detach().cpu()
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)  # 归一化到0-1
        return t

    vutils.save_image(norm_img(batch_edge), f"{save_debug_dir}/batch_{batch_idx}_edge.png")
    vutils.save_image(norm_img(batch_coarse), f"{save_debug_dir}/batch_{batch_idx}_coarse.png")
    vutils.save_image(norm_img(output), f"{save_debug_dir}/batch_{batch_idx}_output.png")
    vutils.save_image(norm_img(batch_label), f"{save_debug_dir}/batch_{batch_idx}_label.png")

# 初始化
optimizer, scheduler = get_optimizer_and_scheduler(model)

epochs = 50
for epoch in tqdm(range(epochs), desc="Training Progress", dynamic_ncols=True):
    model.train()
    epoch_mse = 0
    epoch_psnr = 0
    for batch_idx, (batch_coarse, batch_edge, batch_label) in enumerate(train_loader):        
        t0 = time.time()

        batch_edge, batch_coarse, batch_label = batch_edge.to(device), batch_coarse.to(device), batch_label.to(device)
        #print(batch_edge.device, batch_coarse.device, batch_label.device)
        batch_edge = batch_edge - 128
        batch_coarse = batch_coarse - 128
        batch_label = batch_label - 128
        
        t1 = time.time()
        # 前向
        output = model(batch_edge, batch_coarse)
        # if batch_idx % 10 == 0:
        #     save_debug_images(batch_edge, batch_coarse, output, batch_label, batch_idx)
        #     print('已经保存了，狗熊')
        
        loss = criterion(output, batch_label)
        t2 = time.time()

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t3 = time.time()

        # 计算 PSNR 和 MSE
        if not torch.isfinite(output).all() or not torch.isfinite(batch_label).all():
            print("Warning: NaN or Inf detected in output or labels!")
            print("Output stats:", output.min().item(), output.max().item(), output.mean().item())
            print("Label stats:", batch_label.min().item(), batch_label.max().item(), batch_label.mean().item())

            # 打印前几个值（方便观察是否过大）
            print("Sample output values:", output.view(-1)[:10])
            print("Sample label values:", batch_label.view(-1)[:10])

        psnr, mse = calc_psnr(output.detach(), batch_label)
        if batch_idx % 30 == 0:
            print(f"iter MSE loss: {mse:.6f}")
        epoch_mse += mse
        epoch_psnr += psnr
        # 每 50 step 记录到 TensorBoard
        global_step = epoch * len(train_loader) + batch_idx
        if batch_idx % 50 == 0:
            writer.add_scalar('train/mse', mse, global_step)
            writer.add_scalar('train/psnr', psnr, global_step)
        t4 = time.time()
        #if batch_idx % 20 == 0:
        #    print(f"[Batch {batch_idx}] Data: {t1-t0:.3f}s | Forward: {t2-t1:.3f}s | Backward: {t3-t2:.3f}s | Log: {t4-t3:.3f}s", flush=True)
    scheduler.step()
        # 打印本 epoch 的平均指标
    print(f"Epoch {epoch+1}/{epochs} "
          f"- MSE: {epoch_mse/len(train_loader):.6f} "
          f"- PSNR: {epoch_psnr/len(train_loader):.2f} dB "
          f"- LR: {scheduler.get_last_lr()[0]:.6f}")
    
torch.save(model.state_dict(), "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/models/guided_filter_final.pth")
writer.close()


