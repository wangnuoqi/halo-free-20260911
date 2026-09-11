import os
import numpy as np
import torch
from PIL import Image
##########
# 将同一个文件夹中的两个不同后缀的output 计算psnr

# 你已经有的 PSNR 函数
def calc_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    psnr = 100 if mse == 0 else 10 * np.log10(1.0 / mse)
    return psnr, mse

# 读图片 -> tensor，范围 [0,1]
def load_image_as_tensor(path):
    img = Image.open(path)
    arr = np.array(img).astype(np.float32)

    # 如果是 0~255 的图，归一化到 0~1
    if arr.max() > 1.0:
        arr /= 255.0

    t = torch.from_numpy(arr)

    # [H, W] -> [1, H, W]，灰度图
    if t.ndim == 2:
        t = t.unsqueeze(0)
    # [H, W, C] -> [C, H, W]，彩色图
    elif t.ndim == 3:
        t = t.permute(2, 0, 1)

    return t


# ===== 配置部分 =====
#folder = "/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/down_sample_2_test_log_filtered_origin"  # TODO: 改成你的文件夹路径
folder = "/mnt/d/nuoqi/S2R-HDR/data/Single-Image-HDR/comp" 

# 文件名后缀，根据你的实际情况改
lr_suffix = "-input.tiff"
wls_suffix = ".tiff-WLS-025000.tiff"

# ===== 收集文件路径 =====
lr_images = {}
wls_images = {}

for name in os.listdir(folder):
    path = os.path.join(folder, name)
    if not os.path.isfile(path):
        continue

    if name.endswith(lr_suffix):
        prefix = name[:-len(lr_suffix)]   # 去掉后缀，得到前缀数字
        lr_images[prefix] = path
    elif name.endswith(wls_suffix):
        prefix = name[:-len(wls_suffix)]
        wls_images[prefix] = path

# 找到共同的前缀（成对存在的图）
common_prefixes = sorted(set(lr_images.keys()) & set(wls_images.keys()))

print(f"发现成对图片数量: {len(common_prefixes)}")

total_psnr = 0.0
total_mse = 0.0
count = 0

for prefix in common_prefixes:
    lr_path = lr_images[prefix]
    wls_path = wls_images[prefix]

    pred = load_image_as_tensor(lr_path)     # 预测图（_lr）
    target = load_image_as_tensor(wls_path)  # GT 图（-WLS-025000）

    # 确保尺寸一样
    if pred.shape != target.shape:
        print(f"[跳过] 尺寸不匹配: {prefix}, pred={pred.shape}, target={target.shape}")
        continue

    psnr, mse = calc_psnr(pred, target)
    print(f"{prefix}: PSNR={psnr:.2f} dB, MSE={mse:.6f}")

    total_psnr += psnr
    total_mse += mse
    count += 1

if count > 0:
    avg_psnr = total_psnr / count
    avg_mse = total_mse / count
    print(f"\n平均 PSNR: {avg_psnr:.2f} dB")
    print(f"平均 MSE:  {avg_mse:.6f}")
else:
    print("没有找到有效的成对图片。")
