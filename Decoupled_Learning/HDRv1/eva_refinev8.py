import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset
# from train_refinev6 import EdgeAwareRefineNet
import torch.nn.functional as F
import imageio

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
class EdgeAwareRefineNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        """
        in_channels = 3 (Raw + Coarse + EdgeMap)
        结构：4层卷积的小网络
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, 1, kernel_size=1, bias=False)
        )

    def forward(self, x_raw, x_coarse):
        """
        x_raw:    (B, 1, H, W) - 带有丰富细节但有噪声的原始图
        x_coarse: (B, 1, H, W) - 经过前置滤波的干净图（但边缘可能糊了）
        """
        g, _, _ = compute_sobel_grad(x_coarse, normalize=True)
        inp = torch.cat([x_raw, x_coarse, g], dim=1) 
        delta = self.net(inp)
        out = x_coarse + delta      
        return out
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# syn data s2r
# origin_input_test = '/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/input_test_log'
# input_folder_test = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/result/df_output_s2r'
# input_folder_test_label = '/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/testlabel_log'

# real world data
# origin_input_test = '/mnt/d/nuoqi/S2R-HDR/data/test_classic_pics/testinput'
# input_folder_test = "/mnt/d/nuoqi/S2R-HDR/data/test_classic_pics/df_output"               # 高分辨率输入
# input_folder_test_label = "/mnt/d/nuoqi/S2R-HDR/data/test_classic_pics/testlabel" 

# Single HDR output
# origin_input_test       = '/mnt/d/nuoqi/S2R-HDR/data/Single-Image-HDR/testinput'
# input_folder_test       = '/mnt/d/nuoqi/S2R-HDR/data/Single-Image-HDR/df_output_singlehdr'
# input_folder_test_label = '/mnt/d/nuoqi/S2R-HDR/data/Single-Image-HDR/testlabel'

# Lavel Outdoor output
# origin_input_test       = '/mnt/d/nuoqi/S2R-HDR/data/test_laveloutdoor_hdr/testinput'
# input_folder_test       = '/mnt/d/nuoqi/S2R-HDR/data/test_laveloutdoor_hdr/df_output'
# input_folder_test_label = '/mnt/d/nuoqi/S2R-HDR/data/test_laveloutdoor_hdr/testlabel'

# VIVO data
origin_input_test = '/mnt/d/nuoqi/S2R-HDR/data/test_vivo/testinput'
input_folder_test = '/mnt/d/nuoqi/S2R-HDR/data/test_vivo/df_output'
input_folder_test_label = '/mnt/d/nuoqi/S2R-HDR/data/test_vivo/testlabel'

#prepare for refine circle input  (mixed data) 
# origin_input_test = '/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/traininput'
# input_folder_test = "/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/traininput_refine"               # 高分辨率输入
# input_folder_test_label = "/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/trainlabel" 


#model_path = "/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/models_retrain/refine_filter_best_v6.pth"
model_path = "/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/models_retrain/refine_filter_best_mix_v11.pth"
#model_path = "/mnt/d/nuoqi/halo-free/Decoupled_Learning/HDRv1/models/refine_filter_best_log_v8.pth"

# 输出文件夹
output_dir = "/mnt/d/nuoqi/S2R-HDR/data/test_vivo/rf1stage_output"  # 前一个代表用的数据，后一个表示用的模型（是否origin）
os.makedirs(output_dir, exist_ok=True)

model = EdgeAwareRefineNet(in_channels=3) 
model = nn.DataParallel(model).to(device)

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)     

model.eval()
# 归一化图像保存
def save_float32_tiff(img, save_path):
    """
    直接保存原始 img 为 float32 tiff，不做归一化。
    保留负值和原始动态范围。
    """
    img = img.detach().cpu().numpy()
    # 转为 numpy，去 batch/channel 维度
    if img.ndim == 4:       # [B,C,H,W]
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 1:  # [1,H,W]
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 3:  # [3,H,W] → [H,W,3]
        img = np.transpose(img, (1, 2, 0))
    # [H,W,C]
    img = img.astype(np.float32)
    imageio.imwrite(save_path, img)

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

# 计算Sobel梯度
# --------- 工具：Sobel 梯度幅值（对灰度图）---------
def compute_sobel_grad(x, eps=1e-6, normalize=False):
    # x: Bx1xHxW (若不是，会自动做灰度)
    #x = to_gray(x)
    sobel_x = torch.tensor([[[[-1., 0., 1.],
                              [-2., 0., 2.],
                              [-1., 0., 1.]]]], device=x.device, dtype=x.dtype)
    sobel_y = torch.tensor([[[[-1., -2., -1.],
                              [ 0.,  0.,  0.],
                              [ 1.,  2.,  1.]]]], device=x.device, dtype=x.dtype)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    g  = torch.sqrt(gx*gx + gy*gy + eps)
    if normalize:
        g = g / (g.amax(dim=(2,3), keepdim=True) + eps)
    return g, gx, gy  # Bx1xHxW


def edge_guided_loss(pred, input, target,
                     thresh_mode='percentile',  # 'fixed' or 'percentile'
                     thresh=0.6,               # fixed 时用；或 percentile 的分位，如 0.9
                     dilate=1,                 # 边缘膨胀半径（像素），0 表示不膨胀
                     loss_type='l1',           # 'l1' | 'l2' | 'huber'
                     eps=1e-6,
                     normalize_grad=True,
                     filename=None):
    """
    pred, target: BxCxHxW，取值范围不限（建议都已归一到稳定范围）
    thresh_mode:
        - 'fixed'       : 使用固定阈值（若 normalize_grad=True，则阈值以[0,1]为尺度）
        - 'percentile'  : 每张图按分位数自适应阈值（推荐），thresh取0~1之间的分位比如0.9
    dilate: >0 会用 max-pool 膨胀边缘，扩大约束区域
    """
    assert pred.shape == target.shape, "pred/target 形状必须一致"
    # 1) 计算 label 的梯度（决定“哪里是边缘”）
    g = compute_sobel_grad(input, eps=eps, normalize=normalize_grad)  # Bx1xHxW

    # 2) 生成边缘掩码
    if thresh_mode == 'fixed':
        T = torch.as_tensor(thresh, device=pred.device, dtype=pred.dtype)
        mask = (g > T).float()
    elif thresh_mode == 'percentile':
        # 每张图按分位数得到阈值（更鲁棒）
        # g: Bx1xHxW -> BxN
        q = torch.quantile(g.flatten(1), torch.as_tensor(thresh, device=g.device), dim=1)
        T = q.view(-1, 1, 1, 1)  # Bx1x1x1
        mask = (g > T).float()
    else:
        raise ValueError("thresh_mode must be 'fixed' or 'percentile'")

    # 3) 可选：扩边（膨胀），让边缘周围也参与约束
    if dilate and dilate > 0:
        k = dilate*2 + 1
        mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=dilate)
    # ---------- 保存用于选择像素点的mask --------- #
    # save_dir = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRV1/model_testoutput/mask_0.9'
    # os.makedirs(save_dir, exist_ok=True)
    mask_np = mask[0,0] # 取第一张的第一通道
    #imageio.imwrite(os.path.join(save_dir, 'mask.png'), (mask_np*255).astype('uint8'))
    # save_float32_tiff(mask_np, f"{save_dir}/{filename}_mask.tiff")

    # 4) 计算 masked loss（仅在边缘上）
    diff = pred - target
    if loss_type == 'l1':
        per = diff.abs()
    elif loss_type == 'l2':
        per = diff.pow(2)
    elif loss_type == 'huber':
        per = F.smooth_l1_loss(pred, target, reduction='none')
    else:
        raise ValueError("loss_type must be 'l1' | 'l2' | 'huber'")

    # 对所有通道都使用同一份边缘掩码（来自灰度边缘）
    per = per * mask  # mask: Bx1xHxW，会广播到 C 通道

    denom = mask.sum() * pred.size(1) + eps  # 每通道都计数，防止边缘太少导致偏大
    loss = per.sum() / denom

    # 方便你调参：返回一些统计量
    stats = {
        "edge_ratio": (mask.mean()).item(),           # 边缘像素比例
        "thresh_used": (T.mean().item() if 'T' in locals() else float('nan')),
    }
    return loss, stats

##### - 数据加载 - #######
class GuidedFilterDataset(Dataset):
    def __init__(self, origin_input_dir, input_dir, label_dir):
        self.origin_input_files = sorted([os.path.join(origin_input_dir, f) for f in os.listdir(origin_input_dir)])
        self.input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        print(f"INPUT: {len(self.input_files)}, Label: {len(self.label_files)}")

    def __len__(self):
        return min(len(self.input_files), len(self.label_files))

    def __getitem__(self, idx):

        input_path = self.input_files[idx]
        origin_input   = cv2.imread(self.origin_input_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        input   = cv2.imread(self.input_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        label  = cv2.imread(self.label_files[idx], cv2.IMREAD_UNCHANGED).astype('float32')
        input_hr_h, input_hr_w = input.shape

        filename = os.path.splitext(os.path.basename(input_path))[0]

        return {
            'origin_input_hr': torch.from_numpy(origin_input).unsqueeze(0).unsqueeze(0),
            'input_hr': torch.from_numpy(input).unsqueeze(0).unsqueeze(0),
            'target_img': torch.from_numpy(label).unsqueeze(0).unsqueeze(0),
            'filename': filename,
            'input_height_hr': input_hr_h,
            'input_width_hr': input_hr_w,
        }

def var_custom_collate(batch):
    min_h_hr = min([item['input_height_hr'] for item in batch])
    min_w_hr = min([item['input_width_hr'] for item in batch])

    batch_input = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_input_origin   = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    batch_target_images = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    filenames = []

    for idx, item in enumerate(batch):
        # 直接裁切到 min_h, min_w
        input_origin_crop = item['origin_input_hr'][:, :, :min_h_hr, :min_w_hr]
        input_crop   = item['input_hr'][:, :, :min_h_hr, :min_w_hr]
        label_crop  = item['target_img'][:, :, :min_h_hr, :min_w_hr]

        # 放到 batch
        batch_input_origin[idx] = input_origin_crop
        batch_input[idx] = input_crop
        batch_target_images[idx] = label_crop
        filenames.append(item['filename'])

    return batch_input_origin, batch_input, batch_target_images, filenames

test_dataset = GuidedFilterDataset(
    origin_input_dir = origin_input_test,
    input_dir = input_folder_test,
    label_dir = input_folder_test_label
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                          collate_fn=var_custom_collate, pin_memory=True, drop_last=False)

######### psnr 计算 ###########
def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1 / mse)
    return psnr, mse

import time
test_mse = 0
test_psnr = 0
total_time = 0
total_images = 0
with torch.no_grad():
    for test_input_origin, test_input, test_label, filenames in tqdm(test_loader, desc="Testing"):
        test_input_origin = crop_to_even(test_input_origin)
        test_input = crop_to_even(test_input)
        test_label = crop_to_even(test_label)

        test_input_origin, test_input, test_label = test_input_origin.to(device), test_input.to(device), test_label.to(device)
        
        test_input_origin = test_input_origin - 0.5
        test_input = test_input - 0.5
        test_label = test_label - 0.5

        start_time = time.time()
        test_output = model(test_input_origin, test_input)
        end_time = time.time()
    
        test_output = test_output + 0.5
        test_output = torch.clamp(test_output, 0, 1)
        test_label  = test_label + 0.5

        batch_time = end_time - start_time
        total_time += batch_time
        total_images += test_input.size(0)

        # test_input_masked = test_input_masked + 0.5
        for i, fname in enumerate(filenames):
        #     save_float32_tiff(test_input_masked[i], f"{output_dir}/{fname}_origin_edge.tiff")
            save_float32_tiff(test_output[i], f"{output_dir}/{fname}_pred.tiff")


        psnr, mse = calc_psnr(test_output.detach(), test_label)
        #print(f"\n PSNR: {psnr:.2f} dB")
        test_mse += mse
        test_psnr += psnr

    avg_test_mse = test_mse / len(test_loader)
    avg_test_psnr = test_psnr / len(test_loader)
    avg_infer_time = total_time / total_images

print(f"Test Finished - Average PSNR: {avg_test_psnr:.2f} dB, Average MSE: {avg_test_mse:.6f}")
print(f"Average inference time per image: {avg_infer_time * 1000:.2f} ms")
















