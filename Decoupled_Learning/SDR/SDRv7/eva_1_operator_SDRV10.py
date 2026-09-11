import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset

# 引入你训练时用的模块
#from train_1_operator_SDRmodelv10 import DeepGuidedFilter  # 根据文件名改成实际路径
from train_1_operator_SDRv16 import DeepGuidedFilter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 测试集路径（按你代码里设定）
downsample_folder_test = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/down_sample_2_lanczos_test"  # 低分辨率输入
input_folder_test = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/input_V4"               # 高分辨率输入
label_test = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/WLS-021544_V4"                   # 标签

# 训练集路径 
# downsample_folder_test = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/down_sample_2_lanczos_train"  # 低分辨率输入
# input_folder_test = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRinput"               # 高分辨率输入
# label_test = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRinput_label" 

# 模型文件
model_path = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/models/convguided_filter_best_v16_r1_lanczos.pth"

# 输出文件夹
output_dir = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/refine_img_train"
os.makedirs(output_dir, exist_ok=True)

# 加载模型
model = DeepGuidedFilter(radius=1, eps=1e-8) ##### 根据训练 r 的不同进行调整
model = nn.DataParallel(model).to(device)  # 先包 DataParallel

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)          # 再加载参数

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

class GuidedFilterDataset(Dataset):
    def __init__(self, x_lr_dir, x_hr_dir, label_dir):
        self.x_lr_files = sorted([os.path.join(x_lr_dir, f) for f in os.listdir(x_lr_dir)])
        self.x_hr_files = sorted([os.path.join(x_hr_dir, f) for f in os.listdir(x_hr_dir)])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        print(f"LR: {len(self.x_lr_files)}, HR: {len(self.x_hr_files)}, Label: {len(self.label_files)}")

    def __len__(self):
        return min(len(self.x_lr_files), len(self.x_hr_files), len(self.label_files))

    def __getitem__(self, idx):
        x_lr_path = self.x_lr_files[idx]
        x_hr_path = self.x_hr_files[idx]
        label_path = self.label_files[idx]

        x_lr   = cv2.imread(x_lr_path, cv2.IMREAD_GRAYSCALE).astype('float32')
        x_hr   = cv2.imread(x_hr_path, cv2.IMREAD_GRAYSCALE).astype('float32')
        label  = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype('float32')
        input_hr_h, input_hr_w = x_hr.shape
        input_lr_h, input_lr_w = x_lr.shape

        filename = os.path.splitext(os.path.basename(x_hr_path))[0]  # 用 HR 图的文件名作为基准

        return {
            'input_lr': torch.from_numpy(x_lr).unsqueeze(0).unsqueeze(0),
            'input_hr': torch.from_numpy(x_hr).unsqueeze(0).unsqueeze(0),
            'target_img': torch.from_numpy(label).unsqueeze(0).unsqueeze(0),
            'filename': filename,
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
    batch_x_lr = torch.zeros(len(batch), 1, min_h_lr, min_w_lr)
    batch_target_images = torch.zeros(len(batch), 1, min_h_hr, min_w_hr)

    filenames = []

    for idx, item in enumerate(batch):
        x_lr_crop = item['input_lr'][:, :, :min_h_lr, :min_w_lr]
        x_hr_crop = item['input_hr'][:, :, :min_h_hr, :min_w_hr]
        label_crop = item['target_img'][:, :, :min_h_hr, :min_w_hr]

        batch_x_lr[idx] = x_lr_crop
        batch_x_hr[idx] = x_hr_crop
        batch_target_images[idx] = label_crop
        filenames.append(item['filename'])

    return batch_x_lr, batch_x_hr, batch_target_images, filenames

# 加载测试集
test_dataset = GuidedFilterDataset(
    x_lr_dir=downsample_folder_test,
    x_hr_dir=input_folder_test,
    label_dir=label_test
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                         collate_fn=var_custom_collate, pin_memory=True)

# 评估指标函数
def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1.0 / mse)
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
    t = np.clip(t, 0, 1)

    # 转为 [0,255] uint8
    t = (t * 255.0).round().astype(np.uint8)

    # 保存为 PNG
    cv2.imwrite(path, t)

# 开始测试
import time
total_psnr, total_mse = 0, 0
total_time = 0
total_images = 0
with torch.no_grad():
    for batch_x_lr, batch_x_hr, batch_label, filenames in tqdm(test_loader, desc="Testing"):
        
        batch_x_hr = crop_to_even(batch_x_hr)
        batch_label = crop_to_even(batch_label)
        
        batch_x_lr, batch_x_hr, batch_label = batch_x_lr.to(device), batch_x_hr.to(device), batch_label.to(device)

        batch_x_lr = batch_x_lr - 128
        batch_x_hr = batch_x_hr - 128
        batch_label = batch_label / 255.0

        # 推理
        start_time = time.time()
        output = model(batch_x_lr, batch_x_hr)
        torch.cuda.synchronize()  # 若使用GPU，确保所有CUDA操作完成
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        total_images += output.size(0)
        
        # for i, fname in enumerate(filenames):
        #     norm_and_save(output[i], f"{output_dir}/{fname}_pred.png")
        # 计算指标
        psnr, mse = calc_psnr(output, batch_label)
        total_psnr += psnr
        total_mse += mse

        # 保存结果图像
        #norm_and_save(output, f"{output_dir}/output_{idx}.png")
        #norm_and_save(batch_label, f"{output_dir}/label_{idx}.png")
        #norm_and_save(batch_x_hr, f"{output_dir}/input_hr_{idx}.png")

    # 计算平均指标
    avg_psnr = total_psnr / len(test_loader)
    avg_mse = total_mse / len(test_loader)
    avg_time_per_image = total_time / total_images


print(f"Test Finished - Average PSNR: {avg_psnr:.2f} dB, Average MSE: {avg_mse:.6f}")
print(f"Average Inference Time per Image: {avg_time_per_image * 1000:.2f} ms")
