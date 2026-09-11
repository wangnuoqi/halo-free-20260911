import os
from PIL import Image
import numpy as np
import cv2
import imageio
import time

import torch
'''
  - 读取数据：从 folder_path 里读所有 .tif/.tiff，用 imageio.imread 读成 float32。
  - 统计每张图：
      - 均值、标准差、最小值、最大值
      - 5/50/95 分位数
      - “暗部占比”= 像素 < 0.1 的比例
  - 汇总输出：把所有图的统计指标做 min/max/mean 汇总打印。
'''
folder_path = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRV1/input_test_log'
#folder_path = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRtest_label'
save_folder_path = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRV1'
file_list = [file for file in os.listdir(folder_path) if file.lower().endswith(('.tif', '.tiff'))]
#file_list = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png'))]
file_list.sort() 
images = []
times = []

def hist_equalize_float(img):
    # img: 输入的float32灰度图，shape: (H, W)
    img_flat = img.flatten()
    img_min = img_flat.min()
    img_max = img_flat.max()
    # 归一化到0-1
    img_norm = (img_flat - img_min) / (img_max - img_min)
    # 计算直方图和累积直方图
    hist, bins = np.histogram(img_norm, bins=256, range=[0,1], density=True)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]  # 归一化
    # 用CDF把像素值均匀化
    img_eq = np.interp(img_norm, bins[:-1], cdf)
    # 你可以把它缩放回你要的动态范围
    img_eq = img_eq.reshape(img.shape)
    return img_eq

# def hist_equalize_float_gpu(img_np):
#     import torch
#     img = torch.from_numpy(img_np).float().cuda()
#     img_flat = img.flatten()
#     img_min = img_flat.min()
#     img_max = img_flat.max()
#     img_norm = (img_flat - img_min) / (img_max - img_min + 1e-8)
#     hist_bins = 256
#     hist = torch.histc(img_norm, bins=hist_bins, min=0.0, max=1.0)
#     cdf = torch.cumsum(hist, dim=0)
#     cdf = cdf / cdf[-1]
#     bins = torch.linspace(0, 1, hist_bins, device=img.device)  # length=256

#     img_norm_cpu = img_norm.cpu().numpy()
#     bins_cpu = bins.cpu().numpy()  # 不要[:-1]，直接全部
#     cdf_cpu = cdf.cpu().numpy()
#     img_eq_cpu = np.interp(img_norm_cpu, bins_cpu, cdf_cpu)

#     img_eq = torch.from_numpy(img_eq_cpu).reshape(img.shape).cpu().numpy()
#     return img_eq


means, stds, mins, maxs = [], [], [], []
low_ratio_list = []   # 比如亮度 < 0.1 的像素比例
percentiles = []
for file in file_list:
    file_path = os.path.join(folder_path, file)

    ## PIL 读取 tiff 文件
    # img = Image.open(file_path)
    # arr = np.array(img)

    ## cv2 读取 tiff 文件
    #img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    ## 用imageio读取 tiff 文件
    img = imageio.imread(file_path).astype(np.float32)
    #img = imageio.imread('vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRV1/down_sample_2_train/202411121012180_NormalizedHDR_30000.tiff').astype(np.float32)
    start = time.time()
    #img_eq = hist_equalize_float(img)
    # 扁平化用于统计
    flat = img.flatten()

    # 统计指标
    means.append(flat.mean())
    stds.append(flat.std())
    mins.append(flat.min())
    maxs.append(flat.max())
    percentiles.append(np.percentile(flat, [5, 50, 95]))

    low_ratio = np.mean(flat < 0.1)  # <0.1 亮度的像素占比
    low_ratio_list.append(low_ratio)

    end = time.time()
    run_time = end - start
    run_time_ms = run_time * 1000
    print(f'{file}: 均衡化耗时 {run_time_ms:.2f} 毫秒')
    images.append(img)
    save_path = os.path.join(save_folder_path, file)
    #imageio.imwrite(save_path, img_eq.astype(np.float32))  # 这里把图片对象存到 images 列表里
# 汇总并打印
means = np.array(means)
stds = np.array(stds)
mins = np.array(mins)
maxs = np.array(maxs)
percentiles = np.array(percentiles)
low_ratio_list = np.array(low_ratio_list)

print("\n--- HDR 图像统计指标 ---")
print(f"均值范围：      {means.min():.4f} ~ {means.max():.4f}，平均值：{means.mean():.4f}")
print(f"标准差范围：    {stds.min():.4f} ~ {stds.max():.4f}，平均值：{stds.mean():.4f}")
print(f"最小值范围：    {mins.min():.6f} ~ {mins.max():.6f}")
print(f"最大值范围：    {maxs.min():.4f} ~ {maxs.max():.4f}")
print(f"暗部占比（<0.1）：{low_ratio_list.mean() * 100:.2f}% 平均")

p5, p50, p95 = percentiles[:,0], percentiles[:,1], percentiles[:,2]
print(f"5%分位数范围：  {p5.min():.4f} ~ {p5.max():.4f}，平均：{p5.mean():.4f}")
print(f"50%分位数范围： {p50.min():.4f} ~ {p50.max():.4f}，平均：{p50.mean():.4f}")
print(f"95%分位数范围： {p95.min():.4f} ~ {p95.max():.4f}，平均：{p95.mean():.4f}")
############## in GPU ################
# for file in file_list:
#     file_path = os.path.join(folder_path, file)
#     img = imageio.imread(file_path).astype('float32')
#     start = time.time()
#     img_eq = hist_equalize_float_gpu(img)
#     end = time.time()
#     run_time = end - start
#     run_time_ms = run_time * 1000
#     print(f'{file}: GPU均衡化耗时 {run_time_ms:.2f} 毫秒')
#     images.append(img_eq)
#     times.append(run_time_ms)

