import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset
from train_1_operator_SDRv16_refinev1 import EdgeAwareRefineNet
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_folder_test = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/refine_img_test"               # 高分辨率输入
input_folder_test_label = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/WLS-021544_V4" 


model_path = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/models/refine_filter_best_v2.pth"

# 输出文件夹
output_dir = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/model_test/refine_v2"
os.makedirs(output_dir, exist_ok=True)

model = EdgeAwareRefineNet(in_channels=1) ##### 根据训练 r 的不同进行调整
model = nn.DataParallel(model).to(device)

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)     

model.eval()

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

def compute_sobel_grad(img):
    sobel_x = torch.tensor([[[-1., 0., 1.], 
                                [-2., 0., 2.], 
                                [-1., 0., 1.]]], device=img.device).unsqueeze(0)
    sobel_y = torch.tensor([[[-1., -2., -1.], 
                                [0., 0., 0.], 
                                [1., 2., 1.]]], device=img.device).unsqueeze(0)

    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)
    grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    return grad

def edge_sensitive_loss(pred, target, edge_weight=10.0):
    # 梯度图作为权重
    target_grad = compute_sobel_grad(target)
    weight_map = 1 + edge_weight * target_grad  # 在边缘区域赋予更大的权重

    # 带权重的MSE损失
    loss = ((pred - target) ** 2) * weight_map
    return loss.mean()

##### - 数据加载 - #######
class GuidedFilterDataset(Dataset):
    def __init__(self, input_dir, label_dir):
        #self.x_lr_files = sorted([os.path.join(x_lr_dir, f) for f in os.listdir(x_lr_dir)])
        self.input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        print(f"INPUT: {len(self.input_files)}, Label: {len(self.label_files)}")

    def __len__(self):
        return min(len(self.input_files), len(self.label_files))

    def __getitem__(self, idx):

        input_path = self.input_files[idx]
        #x_lr   = cv2.imread(self.x_lr_files[idx], cv2.IMREAD_GRAYSCALE).astype('float32')
        input   = cv2.imread(self.input_files[idx], cv2.IMREAD_GRAYSCALE).astype('float32')
        label  = cv2.imread(self.label_files[idx], cv2.IMREAD_GRAYSCALE).astype('float32')
        input_hr_h, input_hr_w = input.shape

        filename = os.path.splitext(os.path.basename(input_path))[0]

        return {
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
    #batch_x_lr   = torch.zeros(len(batch), 1, min_h_lr, min_w_lr)
    batch_target_images = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)
    filenames = []

    for idx, item in enumerate(batch):
        # 直接裁切到 min_h, min_w
        #x_lr_crop = item['input_lr'][:, :, :min_h_lr, :min_w_lr]
        input_crop   = item['input_hr'][:, :, :min_h_hr, :min_w_hr]
        label_crop  = item['target_img'][:, :, :min_h_hr, :min_w_hr]

        # 放到 batch
        #batch_x_lr[idx] = x_lr_crop
        batch_input[idx] = input_crop
        batch_target_images[idx] = label_crop
        filenames.append(item['filename'])

    return batch_input, batch_target_images, filenames

test_dataset = GuidedFilterDataset(
    input_dir = input_folder_test,
    label_dir = input_folder_test_label
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                          collate_fn=var_custom_collate, pin_memory=True, drop_last=False)

######### psnr 计算 ###########
def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(255.0 * 255.0 / mse)
    return psnr, mse

# 归一化图像保存
def norm_and_save(tensor, path):
    # 转为 CPU numpy 数组
    t = tensor.detach().cpu().numpy()
    # 去掉 batch/channel 维度 (支持单通道图像)
    if t.ndim == 4:  # [B,C,H,W]
        t = t[0]
    if t.ndim == 3 and t.shape[0] == 1:  # [1,H,W]
        t = t[0]

    # 确保在 [0,1] 范围内
    t = np.clip(t, 0, 255)

    # 转为 [0,255] uint8
    #t = (t * 255.0).round().astype(np.uint8)
    t =  t .round().astype(np.uint8)

    # 保存为 PNG
    cv2.imwrite(path, t)

import time
test_mse = 0
test_psnr = 0
total_time = 0
total_images = 0
with torch.no_grad():
    for test_input, test_label, filenames in tqdm(test_loader, desc="Testing"):
        #test_x_lr = crop_to_even(test_x_lr)
        test_input = crop_to_even(test_input)
        test_label = crop_to_even(test_label)

        test_input, test_label = test_input.to(device), test_label.to(device)
        #test_x_lr = test_x_lr - 128
        test_input = test_input - 128
        test_label = test_label - 128

        torch.cuda.synchronize()
        start_time = time.time()
        test_output = model(test_input)
        torch.cuda.synchronize()
        end_time = time.time()
        
        test_output = test_output + 128
        batch_time = end_time - start_time
        total_time += batch_time
        total_images += test_input.size(0)

        # for i, fname in enumerate(filenames):
        #     norm_and_save(test_output[i], f"{output_dir}/{fname}_pred.png")
        #test_output = model(test_input)

        loss = edge_sensitive_loss(test_output, test_label, edge_weight=10.0)
        #loss = edge_position_loss(test_output, test_label, grad_thresh=0.1)

        psnr, mse = calc_psnr(test_output, test_label)
        test_mse += mse
        test_psnr += psnr

    avg_test_mse = test_mse / len(test_loader)
    avg_test_psnr = test_psnr / len(test_loader)
    avg_infer_time = total_time / total_images

print(f"Test Finished - Average PSNR: {avg_test_psnr:.2f} dB, Average MSE: {avg_test_mse:.6f}")
print(f"Average inference time per image: {avg_infer_time * 1000:.2f} ms")
















