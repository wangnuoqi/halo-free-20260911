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
import argparse
from pathlib import Path

from tqdm import tqdm
try:
    from guided_filter_pytorch.box_filter import BoxFilter
except ImportError:
    BoxFilter = None
#from guided_filter_pytorch.guided_filter import FastGuidedFilter
#from guided_filter_pytorch.guided_filter import GuidedFilter
#from guided_filter_pytorch.guided_filter import ConvGuidedFilter
import cv2
import numpy as np
import time
import math
import random

from dncnn_gray_lr_patch_dataset import DnCNNGrayLRPatchDataset

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

CHANNELS = 1

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

class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1, channels=CHANNELS, norm=nn.BatchNorm2d):
        super(ConvGuidedFilter, self).__init__()
        print("Loaded ConvGuidedFilter from:", __file__)
        self.channels = channels
        self.box_filter = nn.Conv2d(
            channels, channels, kernel_size=3, padding=radius, dilation=radius,
            bias=False, groups=channels
        )
        self.conv_a = nn.Sequential(nn.Conv2d(channels * 2, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, channels, kernel_size=1, bias=False))
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):
        _, c_lrx, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()
        if c_lrx != self.channels:
            raise RuntimeError(f"Expected {self.channels} input channels, got {c_lrx}")

        N = self.box_filter(x_lr.new_ones((1, self.channels, h_lrx, w_lrx)))
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
        if BoxFilter is None:
            raise ImportError("guided_filter_pytorch.box_filter is required to use GuidedFilter")

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

def build_lr_net(norm=AdaptiveNorm, layer=5, channels=CHANNELS):
    layers = [
        nn.Conv2d(channels, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
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

        nn.Conv2d(24, channels, kernel_size=1, stride=1, padding=0, dilation=1)
    ]

    net = nn.Sequential(*layers)

    net.apply(weights_init_identity)

    return net

class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8, channels=CHANNELS):
        super(DeepGuidedFilter, self).__init__()
        self.lr = build_lr_net(channels=channels)
        #self.gf = FastGuidedFilter(radius, eps)
        #self.gf = GuidedFilter(radius, eps)
        self.gf = ConvGuidedFilter(radius, channels=channels, norm=AdaptiveNorm)

    def forward(self, x_lr, x_hr):
        return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)
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
EXPERIMENT_ROOT = str(Path(__file__).resolve().parent)
DEFAULT_TRAIN_DATA_ROOT = r"D:\nuoqi\halo-free\data_bsds\BSDS 400"
DEFAULT_VAL_DATA_ROOT = r"D:\nuoqi\halo-free\data_bsds\Kodak24\PhotoCD_PCD0992"
DEFAULT_TEST_DATA_ROOT = r"D:\nuoqi\halo-free\data_bsds\CBSD68"


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvGuidedFilter denoiser on BSDS AWGN data.")
    parser.add_argument("--train-data-root", default=DEFAULT_TRAIN_DATA_ROOT)
    parser.add_argument("--val-data-root", default=DEFAULT_VAL_DATA_ROOT)
    parser.add_argument("--test-data-root", default=DEFAULT_TEST_DATA_ROOT)
    parser.add_argument("--experiment-root", default=EXPERIMENT_ROOT)
    parser.add_argument("--task-name", default="convguided_bsds_sigma25")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sigma", type=float, default=25.0)
    parser.add_argument(
        "--patch-size", type=int, default=192,
        help="HR training patch size; 192 gives a 96x96 LR patch for scale=2.",
    )
    parser.add_argument(
        "--patch-stride", type=int, default=20,
        help="Legacy argument; DatasetDnCNN uses one random crop per image and ignores stride.",
    )
    parser.add_argument("--downsample-scale", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


args = parse_args()

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

device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
model =  DeepGuidedFilter(radius=1, eps=1e-8)
model = torch.nn.DataParallel(model).to(device)

##### 构建训练、验证和测试集 ####
train_dataset = DnCNNGrayLRPatchDataset(
    args.train_data_root, sigma=args.sigma, patch_size=args.patch_size,
    scale=args.downsample_scale, train=True, seed=42
)
val_dataset = DnCNNGrayLRPatchDataset(
    args.val_data_root, sigma=args.sigma, patch_size=args.patch_size,
    scale=args.downsample_scale, train=False, seed=43
)
test_dataset = DnCNNGrayLRPatchDataset(
    args.test_data_root, sigma=args.sigma, patch_size=args.patch_size,
    scale=args.downsample_scale, train=False, seed=44
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True, drop_last=False)

criterion = torch.nn.MSELoss(reduction='mean')

##### tensorboard 启动指令 #######
#writer = SummaryWriter(log_dir="/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/log_dir")

task_name = args.task_name
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(args.experiment_root, "log_dir", f"{task_name}_{timestamp}")
os.makedirs(log_dir, exist_ok=True)

writer = SummaryWriter(log_dir=log_dir)

# --- 在每个 epoch 结束后保存 ---
def save_checkpoint(epoch, best_loss, best_train_loss, is_best=False, is_train_best=False):
    # 保存完整状态，方便续训
    full_path = os.path.join(save_dir, f"{task_name}_full.pth")
    #full_path = os.path.join(save_dir, "convguided_filter_full_hdrv1_r1_lanczos.pth") #均衡化
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
        'best_train_loss': best_train_loss,
    }, full_path)
    print(f"Checkpoint saved: {full_path}")

    # 保存纯模型 (方便推理或部署)
    last_model_path = os.path.join(save_dir, f"{task_name}_last.pth")
    #last_model_path = os.path.join(save_dir, "convguided_filter_last_hdrv1_r1_lanczos.pth") #均衡化
    torch.save(model.state_dict(), last_model_path)
    print(f"Last model weights saved: {last_model_path}")

    # 如果是最佳模型，另存一份
    if is_best:
        best_model_path = os.path.join(save_dir, f"{task_name}_best.pth")
        #best_model_path = os.path.join(save_dir, f"convguided_filter_best_hdrv1_r1_lanczos.pth") # 均衡化
        torch.save(model.state_dict(), best_model_path)
        print(f"*** Validation-best model updated: {best_model_path} (Val MSE: {best_loss:.6f}) ***")

    if is_train_best:
        train_best_model_path = os.path.join(save_dir, f"{task_name}_train_best.pth")
        torch.save(model.state_dict(), train_best_model_path)
        print(f"*** Train-best model updated: {train_best_model_path} (Train MSE: {best_train_loss:.6f}) ***")

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
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5) 
best_loss = float("inf")
best_train_loss = float("inf")
save_dir = os.path.join(args.experiment_root, "models", task_name)
os.makedirs(save_dir, exist_ok=True)
epochs = args.epochs
start_epoch = 0

###### 重新加载训练模型 ###########
resume_path = os.path.join(save_dir, f"{task_name}_full.pth")
if args.resume and os.path.exists(resume_path):
    print(f"=> Resuming training from checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)

    # 注意：这里的 model 已经是 DataParallel(model).to(device) 了
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 这里保存的是 epoch+1，所以下一轮就从这个 epoch 开始
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint.get('best_loss', float("inf"))
    best_train_loss = checkpoint.get('best_train_loss', float("inf"))

    print(
        f"=> Resumed from epoch {start_epoch}, "
        f"best_val_mse = {best_loss:.6f}, "
        f"best_train_mse = {best_train_loss:.6f}"
    )
elif args.resume:
    print(f"=> No checkpoint found at {resume_path}, training from scratch.")
else:
    print("=> No checkpoint found, training from scratch.")

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
    #for epoch in tqdm(range(epochs), desc="Training Progress", dynamic_ncols=True):
    for epoch in tqdm(range(start_epoch, epochs), desc="Training Progress", dynamic_ncols=True):
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
        is_train_best = avg_epoch_mse < best_train_loss
        if is_train_best:
            best_train_loss = avg_epoch_mse
            # 打印本 epoch 的平均指标
        print(f"Epoch {epoch+1}/{epochs} "
            f"- MSE: {avg_epoch_mse:.6f} "
            f"- PSNR: {avg_epoch_psnr:.2f} dB ")
        writer.add_scalar('train/mse_epoch', avg_epoch_mse, epoch)
        writer.add_scalar('train/psnr_epoch', avg_epoch_psnr, epoch)
        scheduler.step()

        def evaluate(loader):
            model.eval()
            total_mse = 0
            total_psnr = 0
            with torch.no_grad():
                for eval_x_lr, eval_x_hr, eval_label in loader:
                    eval_x_hr = crop_to_even(eval_x_hr)
                    eval_label = crop_to_even(eval_label)
                    eval_x_lr = eval_x_lr.to(device) - 0.5
                    eval_x_hr = eval_x_hr.to(device) - 0.5
                    eval_label = eval_label.to(device)
                    eval_output = model(eval_x_lr, eval_x_hr)
                    psnr, mse = calc_psnr(eval_output, eval_label)
                    total_mse += mse
                    total_psnr += psnr
            return total_mse / len(loader), total_psnr / len(loader)

        # Validation selects checkpoints; test is reported independently.
        avg_val_mse, avg_val_psnr = evaluate(val_loader)
        avg_test_mse, avg_test_psnr = evaluate(test_loader)
        print(f"Epoch {epoch+1}/{epochs} - Val MSE: {avg_val_mse:.6f} - Val PSNR: {avg_val_psnr:.2f} dB")
        print(f"Epoch {epoch+1}/{epochs} - Test MSE: {avg_test_mse:.6f} - Test PSNR: {avg_test_psnr:.2f} dB")
        writer.add_scalar('val/mse', avg_val_mse, epoch)
        writer.add_scalar('val/psnr', avg_val_psnr, epoch)
        writer.add_scalar('test/mse', avg_test_mse, epoch)
        writer.add_scalar('test/psnr', avg_test_psnr, epoch)

        # 判断是否最佳模型
        is_best = avg_val_mse < best_loss
        if is_best:
            best_loss = avg_val_mse

        # 保存模型
        save_checkpoint(
            epoch,
            best_loss,
            best_train_loss,
            is_best=is_best,
            is_train_best=is_train_best,
        )

    # 关闭 TensorBoard writer
    writer.close()
