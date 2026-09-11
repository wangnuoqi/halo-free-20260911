import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from torch.nn import functional as F

import os, datetime

from tqdm import tqdm
from guided_filter_pytorch.box_filter import BoxFilter
#from guided_filter_pytorch.guided_filter import FastGuidedFilter
#from guided_filter_pytorch.guided_filter import GuidedFilter
#from guided_filter_pytorch.guided_filter import ConvGuidedFilter
import cv2
import numpy as np
import time
import math
import random

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()
        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))
        self.bn  = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)

def weights_init_residual(m):
    classname = m.__class__.__name__
    
    # 1. 处理卷积层
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            m.bias.data.zero_()
            
    # 2. 处理标准的 BatchNorm2d
    elif classname.find('BatchNorm2d') != -1:
        if m.weight is not None:
            nn.init.constant_(m.weight.data, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

    # 3. 单独处理 AdaptiveNorm
    elif classname.find('AdaptiveNorm') != -1:
        # AdaptiveNorm 的目标是初始状态接近 Identity (恒等映射)
        # 所以我们希望：Output = 1.0 * x + 0.0 * BN(x)
        # 虽然你的类定义里已经默认初始化为 1 和 0 了，但为了保险起见，这里显式再设一次
        if hasattr(m, 'w_0'):
            nn.init.constant_(m.w_0, 1.0)
        if hasattr(m, 'w_1'):
            nn.init.constant_(m.w_1, 0.0)

class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1, norm=nn.BatchNorm2d):
        super(ConvGuidedFilter, self).__init__()
        print("Loaded ConvGuidedFilter from:", __file__)
        self.box_filter = nn.Conv2d(1, 1, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=1)
        self.conv_a = nn.Sequential(nn.Conv2d(2, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 1, kernel_size=1, bias=False))
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0))
        ## mean_x
        mean_x = self.box_filter(x_lr)/N
        ## mean_y
        mean_y = self.box_filter(y_lr)/N
        ## cov_xy
        cov_xy = self.box_filter(x_lr * y_lr)/N - mean_x * mean_y
        ## var_x
        var_x  = self.box_filter(x_lr * x_lr)/N - mean_x * mean_x

        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return mean_A * x_hr + mean_b

class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(x.new_ones((1, 1, h_x, w_x)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b

def build_lr_net(norm=AdaptiveNorm, layer=5):
    # ... (前面的层定义保持不变) ...
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

        # 最后一层：输出 Residual Map (1 channel)
        nn.Conv2d(24, 1, kernel_size=1, stride=1, padding=0, dilation=1)
    ]

    net = nn.Sequential(*layers)

    # === 修改这里：使用标准初始化，而不是 Identity 初始化 ===
    net.apply(weights_init_residual) 
    
    # 这是一个小技巧：显式地将最后一层的权重缩小
    # 这样网络一开始预测的残差几乎为0，也就是 Output ≈ Input
    # 这对去噪/滤波任务是极好的起点
    net[-1].weight.data.mul_(0.1) 
    if net[-1].bias is not None:
        net[-1].bias.data.zero_()

    return net

class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8):
        super(DeepGuidedFilter, self).__init__()
        self.lr = build_lr_net()
        #self.gf = FastGuidedFilter(radius, eps)
        #self.gf = GuidedFilter(radius, eps)
        self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm)

    def forward(self, x_lr, x_hr):
        residual_lr = self.lr(x_lr)
        residual_hr = self.gf(x_lr, residual_lr, x_hr)
        return (x_hr + residual_hr).clamp(0, 1)
        #return self.gf(x_hr, self.lr(x_hr)).clamp(0, 1)

    def init_lr(self, path):
        self.lr.load_state_dict(torch.load(path))

#### - 需要输入网络的基础信息图，用于推理得到真实的滤波图 - ####
#### - 训练过程中，无论我们的输入如何变化，我们的label都保持不变。是归一化到 0-1 后的
# ##### 训练输入 from HDR512_input #####
# downsample_folder_train = "/mnt/d/nuoqi/S2R-HDR/data/lavelindoor_hdr/down_sample_2_train" ## yes
# input_folder_train = '/mnt/d/nuoqi/S2R-HDR/data/lavelindoor_hdr/traininput'

# label_train = "/mnt/d/nuoqi/S2R-HDR/data/lavelindoor_hdr/trainlabel" ## yes

# ####### - 测试输入 from HDR512_valinput - #########
# downsample_folder_test = '/mnt/d/nuoqi/S2R-HDR/data/lavelindoor_hdr/down_sample_2_test'
# input_folder_test = '/mnt/d/nuoqi/S2R-HDR/data/lavelindoor_hdr/testinput'

# label_test = '/mnt/d/nuoqi/S2R-HDR/data/lavelindoor_hdr/testlabel'
downsample_folder_train = "/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/down_sample_2_traininput"
input_folder_train = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/traininput'
label_train = "/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/trainlabel"

downsample_folder_test = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/down_sample_2_testinput'
input_folder_test = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/testinput'
label_test = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/testlabel'

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
        x_lr   = cv2.imread(self.x_lr_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        x_hr   = cv2.imread(self.x_hr_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        label  = cv2.imread(self.label_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
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

############ 处理数据附带的部分函数 #############
# ------- crop 文件避免产生奇数影响采样和输出 ---------------#
def crop_to_even(tensor):
    """
    将输入 Tensor 裁剪到高宽为偶数，保持左上角对齐（裁掉底部和右边）。
    tensor: [B, C, H, W]
    返回裁剪后的 tensor
    """
    _, _, H, W = tensor.shape
    new_H = H if H % 2 == 0 else H - 1
    new_W = W if W % 2 == 0 else W - 1
    return tensor[:, :, :new_H, :new_W]
#############  对输入的normalized 处理 #############
def normalize_hdr(img, scale=None, mode='P95'):
    if scale is None:
        if mode == 'P95':
            scale = 1.0 / torch.quantile(img, 0.95)
        elif mode == 'max':
            scale = 1.0 / img.max()
        else:
            raise ValueError("Unsupported mode")
    return img * scale, scale

def _to_gray(x: torch.Tensor) -> torch.Tensor:
    # x: (B, C, H, W) 或 (B,1,H,W)
    if x.dim() == 3:  # (B,H,W)
        x = x.unsqueeze(1)
    if x.size(1) == 1:
        return x
    # 简单均值转灰度（需要更严格可以改为加权）
    return x.mean(dim=1, keepdim=True)

def _sobel_grad(x: torch.Tensor, eps: float = 1e-6):
    # 反射填充，边缘更稳
    x = F.pad(x, (1,1,1,1), mode='reflect')
    kx = torch.tensor([[[[-1.,0.,1.],
                         [-2.,0.,2.],
                         [-1.,0.,1.]]]], device=x.device, dtype=x.dtype)
    ky = torch.tensor([[[[-1.,-2.,-1.],
                         [ 0., 0., 0.],
                         [ 1., 2., 1.]]]], device=x.device, dtype=x.dtype)
    gx = F.conv2d(x, kx)
    gy = F.conv2d(x, ky)
    mag = torch.sqrt(gx*gx + gy*gy + eps)
    return gx, gy, mag

def sobel_gradient_loss(pred: torch.Tensor,
                        target: torch.Tensor,
                        mode: str = 'vector',   # 'vector' 或 'magnitude'
                        reduction: str = 'mean',
                        eps: float = 1e-6) -> torch.Tensor:
    """
    pred, target: (B,C,H,W) 或 (B,1,H,W)
    mode='vector'  : L1(|∂x| + |∂y|)（推荐，方向+幅值都对齐）
    mode='magnitude': L1(|∇|)        （仅幅值）
    """
    pred_g = _to_gray(pred)
    targ_g = _to_gray(target)

    pgx, pgy, pm = _sobel_grad(pred_g, eps)
    tgx, tgy, tm = _sobel_grad(targ_g, eps)

    if mode == 'vector':
        loss = (pgx - tgx).abs() + (pgy - tgy).abs()
    elif mode == 'magnitude':
        loss = (pm - tm).abs()
    else:
        raise ValueError("mode must be 'vector' or 'magnitude'")

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError("reduction must be 'mean' | 'sum' | 'none'")

device = torch.device('cuda:0')
model =  DeepGuidedFilter(radius=1, eps=1e-8)
model = torch.nn.DataParallel(model).to(device)

##### 构建训练集和测试集 ####
train_dataset = GuidedFilterDataset(
    x_lr_dir = downsample_folder_train,
    x_hr_dir = input_folder_train,
    label_dir = label_train
)

test_dataset = GuidedFilterDataset(
    x_lr_dir = downsample_folder_test,
    x_hr_dir = input_folder_test,
    label_dir = label_test
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0,
                          collate_fn=var_custom_collate, pin_memory=True, drop_last=True)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0,
                          collate_fn=var_custom_collate, pin_memory=True, drop_last=False)

criterion = torch.nn.MSELoss(reduction='mean')

##### tensorboard 启动指令 #######
#writer = SummaryWriter(log_dir="/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/log_dir")

task_name = "convguided_filter_residual_mix_residual"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/log_dir", f"{task_name}_{timestamp}")

writer = SummaryWriter(log_dir=log_dir)

# --- 在每个 epoch 结束后保存 ---
def save_checkpoint(epoch, best_loss, is_best=False):
    # 保存完整状态，方便续训
    full_path = os.path.join(save_dir, "convguided_filter_residual_full_mix_residual.pth")
    #full_path = os.path.join(save_dir, "convguided_filter_full_hdrv1_r1_lanczos.pth") #均衡化
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss
    }, full_path)
    print(f"Checkpoint saved: {full_path}")

    # 保存纯模型 (方便推理或部署)
    last_model_path = os.path.join(save_dir, "convguided_filter_residual_last_mix_residual.pth")
    #last_model_path = os.path.join(save_dir, "convguided_filter_last_hdrv1_r1_lanczos.pth") #均衡化
    torch.save(model.state_dict(), last_model_path)
    print(f"Last model weights saved: {last_model_path}")

    # 如果是最佳模型，另存一份
    if is_best:
        best_model_path = os.path.join(save_dir, f"convguided_filter_residual_best_mix_residual.pth")
        #best_model_path = os.path.join(save_dir, f"convguided_filter_best_hdrv1_r1_lanczos.pth") # 均衡化
        torch.save(model.state_dict(), best_model_path)
        print(f"*** Best model updated: {best_model_path} (Test MSE: {best_loss:.6f}) ***")

######### psnr 计算 ###########
def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10( 1 / mse)
    return psnr, mse

# def get_optimizer_and_scheduler(model, lr_1=0.01, lr_2=0.001, switch_epoch=0, total_epochs=150):
#     optimizer = optim.Adam(model.parameters(), lr=lr_1)

#     def lr_lambda(epoch):
#         # 前 switch_epoch：返回 1.0（保持 lr=lr_1）
#         # 后面：返回 lr_2 / lr_1（缩小到 0.01）
#         return 1.0 if epoch < switch_epoch else lr_2 / lr_1

#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
#     return optimizer, scheduler

# save_debug_dir = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7"
# os.makedirs(save_debug_dir, exist_ok=True)

# def save_debug_images(batch_x_lr, batch_x_hr, output, batch_label, batch_idx):
#     # 确保是 CPU 和 [0,1] 范围
#     def norm_img(tensor):
#         t = tensor.detach().cpu()
#         t = (t - t.min()) / (t.max() - t.min() + 1e-8)  # 归一化到0-1
#         return t

#     vutils.save_image(norm_img(batch_x_lr), f"{save_debug_dir}/batch_{batch_idx}_ilr.png")
#     vutils.save_image(norm_img(batch_x_hr), f"{save_debug_dir}/batch_{batch_idx}_ihr.png")
#     vutils.save_image(norm_img(output), f"{save_debug_dir}/batch_{batch_idx}_output.png")
#     vutils.save_image(norm_img(batch_label), f"{save_debug_dir}/batch_{batch_idx}_label.png")

# 初始化
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5) 
best_loss = float("inf")
save_dir = "/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/models_retrain"
epochs = 80

###### 重新加载训练模型 ###########
# resume_path = os.path.join(save_dir, "convguided_filter_full_lavel.pth")
# if os.path.exists(resume_path):
#     print(f"=> Resuming training from checkpoint: {resume_path}")
#     checkpoint = torch.load(resume_path, map_location=device)

#     # 注意：这里的 model 已经是 DataParallel(model).to(device) 了
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

#     # 这里保存的是 epoch+1，所以下一轮就从这个 epoch 开始
#     start_epoch = checkpoint['epoch']
#     best_loss = checkpoint.get('best_loss', float("inf"))

#     print(f"=> Resumed from epoch {start_epoch}, best_loss = {best_loss:.6f}")
# else:
#     print("=> No checkpoint found, training from scratch.")

debug_mode = True
if __name__ == "__main__":
    # if debug_mode:
    #     # ====== 只跑测试流程 ======
    #     model.load_state_dict(torch.load('/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRV1/models/convguided_filter_best_hdrv1_r1_lanczos.pth'))  # 替换为你的模型路径
    #     model.eval()
    #     with torch.no_grad():
    #         for test_x_lr, test_x_hr, test_label in test_loader:
    #             test_x_hr = crop_to_even(test_x_hr)
    #             test_label = crop_to_even(test_label)
    #             test_x_lr, test_x_hr, test_label = test_x_lr.to(device), test_x_hr.to(device), test_label.to(device)
    #             test_x_lr -= 0.5
    #             test_x_hr -= 0.5
    #             test_label = test_label / 255
    #             test_label = (test_label - test_label.min()) / (test_label.max() - test_label.min() + 1e-8)

    #             test_output = model(test_x_lr, test_x_hr)
    #             psnr, mse = calc_psnr(test_output, test_label)
    #             print(f"[Debug Test] - MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")
    #     exit()
######## -- 模型训练 -- #############
    seed_everything(42) 
    for epoch in tqdm(range(epochs), desc="Training Progress", dynamic_ncols=True):
    #for epoch in tqdm(range(start_epoch, epochs), desc="Training Progress", dynamic_ncols=True):
        model.train()
        epoch_mse = 0
        epoch_psnr = 0
        for batch_idx, (batch_x_lr, batch_x_hr, batch_label) in enumerate(train_loader): 
            # if batch_idx > 0:  # 只跑第一个 iter
            #     break       

            #batch_x_lr = crop_to_even(batch_x_lr)
            batch_x_hr = crop_to_even(batch_x_hr)
            batch_label = crop_to_even(batch_label)

            batch_x_lr, batch_x_hr, batch_label = batch_x_lr.to(device), batch_x_hr.to(device), batch_label.to(device)
         
            batch_x_lr = batch_x_lr - 0.5
            batch_x_hr = batch_x_hr - 0.5
            batch_label = batch_label
            # 前向
            output = model(batch_x_lr, batch_x_hr)
            #output = inverse_tonemap_mu(output, mu=1000.0)
            #output = model(batch_x_hr)

            # if batch_idx % 10 == 0:
            #     save_debug_images( batch_x_lr, batch_x_hr, output, batch_label, batch_idx)
            #     print('已经保存了，狗熊')
            eps = 1e-8
            loss_mse = criterion(output, batch_label)
            loss_g = sobel_gradient_loss(output, batch_label, mode='vector')      # 推荐
            scale_g = (loss_g / (loss_mse + eps)).detach().clamp_min(eps)
            loss = loss_mse + loss_g / scale_g

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
        avg_epoch_mse = epoch_mse / len(train_loader)
        avg_epoch_psnr = epoch_psnr / len(train_loader)
            # 打印本 epoch 的平均指标
        print(f"Epoch {epoch+1}/{epochs} "
            f"- MSE: {avg_epoch_mse:.6f} "
            f"- PSNR: {avg_epoch_psnr:.2f} dB ")
        writer.add_scalar('train/mse_epoch', avg_epoch_mse, epoch)
        writer.add_scalar('train/psnr_epoch', avg_epoch_psnr, epoch)
        scheduler.step()

 # ---------------- 测试集验证阶段 ----------------
        model.eval()
        test_mse = 0
        test_psnr = 0
        with torch.no_grad():
            for test_x_lr, test_x_hr, test_label in test_loader:
                #test_x_lr = crop_to_even(test_x_lr)
                test_x_hr = crop_to_even(test_x_hr)
                test_label = crop_to_even(test_label)

                test_x_lr, test_x_hr, test_label = test_x_lr.to(device), test_x_hr.to(device), test_label.to(device)
                test_x_lr = test_x_lr - 0.5
                test_x_hr = test_x_hr - 0.5
                test_label = test_label
                # ========== Normalize ==========
                # test_x_lr, scale = normalize_hdr(test_x_lr, mode='P95')
                # test_x_hr, scale = normalize_hdr(test_x_hr, mode='P95')
                # test_label_norm = test_label * scale
                # ========== Normalize ==========
                # test_x_lr = tonemap_mu(test_x_lr, mu=1000.0)
                # test_x_hr = tonemap_mu(test_x_hr, mu=1000.0)
                # test_label = tonemap_mu(test_label, mu=1000.0)

                test_output = model(test_x_lr, test_x_hr)
                #test_output = inverse_tonemap_mu(test_output, mu=1000.0)
                #test_output = model(test_x_hr)

                v_loss = criterion(test_output, test_label)

                psnr, mse = calc_psnr(test_output, test_label)
                test_mse += mse
                test_psnr += psnr

        avg_test_mse = test_mse / len(test_loader)
        avg_test_psnr = test_psnr / len(test_loader)

        print(f"Epoch {epoch+1}/{epochs} "
            f"- Test MSE: {avg_test_mse:.6f} "
            f"- Test PSNR: {avg_test_psnr:.2f} dB")
        # --- 记录到 TensorBoard ---
        writer.add_scalar('test/mse', avg_test_mse, epoch)
        writer.add_scalar('test/psnr', avg_test_psnr, epoch)

        # 判断是否最佳模型
        is_best = avg_test_mse < best_loss
        if is_best:
            best_loss = avg_test_mse

        # 保存模型
        save_checkpoint(epoch, avg_test_mse, is_best=is_best)
 
    # 关闭 TensorBoard writer
    writer.close()


