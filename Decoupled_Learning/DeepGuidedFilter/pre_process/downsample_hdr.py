import os
import cv2
from tqdm import tqdm
import numpy as np
import imageio

################### 对输入做一个下采样和直方图均衡化，所有的输出来自于输入。一定要和我们最终用到模型里的label对应上 #######################
# 输入输出路径
input_dir = "/mnt/d/nuoqi/S2R-HDR/data/test_classic_pics/testinput"
#input_dir = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRV1_data/test_origin"
#input_dir = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRV1_data/input_origin"
#input_dir = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR512_input"
#input_dir_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/input_V4' 

output_dir = "/mnt/d/nuoqi/S2R-HDR/data/test_classic_pics/down_sample_2"
#output_dir = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRV1_data/down_sample_2_test_log"
output_dir_origin = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRV1/input_train_log"
#cleaned_dir = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/halo_free_test'
# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
#os.makedirs(output_dir_origin, exist_ok=True)

#----------- 传统直方图均衡化 ---------- #
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
# -------- clahe_equalize_float均衡化 ---------- # 
def clahe_equalize_float(img):
    # 输入img: float32 0~1
    img_8bit = (img * 255).clip(0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 你可以调参
    img_clahe = clahe.apply(img_8bit)
    img_clahe = img_clahe.astype(np.float32) / 255.0
    return img_clahe

# --------- Retinex均衡化 ------------------ #
def MSR(img, sigmas=[15, 80, 250]):
    img = img.astype(np.float32) + 1e-6
    result = np.zeros_like(img)
    for sigma in sigmas:
        blur = cv2.GaussianBlur(img, (0,0), sigma)
        safe_img = img + 1e-8
        safe_blur = blur + 1e-8
        result += (np.log(img) - np.log(blur))
    result = result / len(sigmas)
    # 归一化到0-1
    result = (result - result.min()) / (result.max() - result.min())
    return result
# ------------自适应均衡化 --------------- #
def adaptive_contrast_stretch(img, low_threshold=0.05, high_threshold=0.90):
    """
    img: float32 灰度图（任意范围）
    low_threshold, high_threshold: 拉伸阈值
    """
    img_min = np.percentile(img, low_threshold*100)
    img_max = np.percentile(img, high_threshold*100)
    dynamic_range = img_max - img_min

    # 你可以改成0.2, 0.5, 按你的实际需求
    if dynamic_range < 0.1:
        # 动态范围太小，全局强力拉伸
        img_stretch = (img - img_min) / (img_max - img_min + 1e-8)
    else:
        # 动态范围已经足够大，轻微拉伸或直接clip
        img_stretch = np.clip((img - img_min) / (img_max - img_min + 1e-8), 0, 1)
    return img_stretch

# ------------ 自适应优化 （改进版）------------- #
def stretch_dark_to_half(img, mean_thresh=0.3, dark_thresh=0.5):
    """
    如果图像均值小于mean_thresh，只对暗区（<dark_thresh）的像素
    做直方图均衡，并让它们均匀分布在[0, dark_thresh]内。
    其他像素保持不变。
    """
    img = np.clip(img, 0, 1)
    if img.mean() >= mean_thresh:
        return img  # 不处理

    mask = img < dark_thresh
    dark_pixels = img[mask]
    if dark_pixels.size == 0:
        return img

    # 拉伸并均匀分布到[0, dark_thresh]内
    # 归一化到[0,1]
    dark_norm = (dark_pixels - dark_pixels.min()) / (dark_pixels.max() - dark_pixels.min() + 1e-8)
    # 直方图均衡化
    hist, bins = np.histogram(dark_norm, bins=256, range=[0,1], density=True)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]
    dark_eq = np.interp(dark_norm, bins[:-1], cdf)
    # 均匀分布拉回[0, dark_thresh]
    stretched = dark_eq * dark_thresh

    img_out = img.copy()
    img_out[mask] = stretched
    return img_out

# --------------- clean filter 代码块 ---------------- #
def fix_towards_jump(I, i, bias=0.7):
    left = I[i - 1]
    right = I[i + 1]
    left_diff = abs(I[i] - left)
    right_diff = abs(I[i] - right)
    return bias * left + (1 - bias) * right if left_diff > right_diff else (1 - bias) * left + bias * right

def remove_halo_1d(I, lap_thresh=25, grad_thresh=5, bias=0.6):
    I = I.astype(np.float32)
    grad = I[1:] - I[:-1]
    grad = np.pad(grad, (0,1), mode='edge')
    lap = np.convolve(I, [1, -2, 1], mode='same')
    mask = (np.abs(lap) > lap_thresh) & (np.abs(grad) < grad_thresh)
    I_filtered = I.copy()
    for i in np.where(mask)[0]:
        if 1 <= i < len(I) - 1:
            I_filtered[i] = fix_towards_jump(I, i, bias)
    return I_filtered

def remove_halo_2d(img, lap_thresh=25, grad_thresh=5, bias=0.6):
    """
    对灰度图像进行 halo 去除（水平、垂直、主对角线、反对角线方向）
    """
    img = img.astype(np.float32)
    h, w = img.shape
    result = img.copy()

    # 水平方向
    for y in range(h):
        result[y, :] = remove_halo_1d(result[y, :], lap_thresh, grad_thresh, bias)

    # 垂直方向
    for x in range(w):
        result[:, x] = remove_halo_1d(result[:, x], lap_thresh, grad_thresh, bias)

    # # 主对角线 ↘︎
    # for k in range(-h + 2, w - 1):  # 避免太短的对角线
    #     diag = np.diagonal(result, offset=k)
    #     if len(diag) >= 3:
    #         fixed = remove_halo_1d(diag, lap_thresh, grad_thresh, bias)
    #         i, j = (0, k) if k >= 0 else (-k, 0)
    #         for idx in range(len(fixed)):
    #             result[i + idx, j + idx] = fixed[idx]

    # # 副对角线 ↙︎（处理翻转后的主对角线）
    # flipped = np.fliplr(result.copy())
    # for k in range(-h + 2, w - 1):
    #     diag = np.diagonal(flipped, offset=k)
    #     if len(diag) >= 3:
    #         fixed = remove_halo_1d(diag, lap_thresh, grad_thresh, bias)
    #         i, j = (0, k) if k >= 0 else (-k, 0)
    #         for idx in range(len(fixed)):
    #             flipped[i + idx, j + idx] = fixed[idx]
    # result = np.fliplr(flipped)

    return np.clip(result, 0, 255).astype(np.uint8)

def save_normalized_float32_tiff(save_path, img):
    """
    将输入img归一化到[0,1]（动态范围最大化），再保存为float32 tiff。
    """
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()
    # 如果max==min，避免除零，直接保存全黑
    if img_max - img_min < 1e-8:
        img_norm = np.zeros_like(img)
    else:
        img_norm = (img - img_min) / (img_max - img_min)
    imageio.imwrite(save_path, img_norm.astype(np.float32))

def save_float32_tiff(save_path, img):
    """
    直接保存原始 img 为 float32 tiff，不做归一化。
    保留负值和原始动态范围。
    """
    img = img.astype(np.float32)
    imageio.imwrite(save_path, img)
##############################################################################################
# 遍历所有 PNG 文件
for filename in tqdm(os.listdir(input_dir), desc="Processing files", unit="file"):
    if filename.lower().endswith(".tiff"):
        file_path = os.path.join(input_dir, filename)

        # 读取图像
        #img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = imageio.imread(file_path).astype(np.float32)
        #img = imageio.imread('vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR512_input/202411121012180_NormalizedHDR_30000.tiff')
        #img_cleaned = remove_halo_2d(img)
        #save_path_cleaned = os.path.join(cleaned_dir, filename)
        #cv2.imwrite(save_path_cleaned, img_cleaned)

        if img is None:
            print(f"无法读取: {file_path}")
            continue

        ###将图像转移到log域内####
        # eps = 1e-8
        # linear_img = np.clip(img, eps, 1.0)
        # img = 1024.0 * np.log(linear_img)
        # img_min = img.min()
        # img_max = img.max()
        # img = (img - img_min) / (img_max - img_min)  # 自然对数

        # 获取原始尺寸
        h, w = img.shape[:2]
        if h % 2 != 0:
            h -= 1
        if w % 2 != 0:
            w -= 1
        img = img[:h, :w]  # 裁剪图像

        # 计算目标尺寸（长边缩小 n(2,4,6,8,...) 倍，保持比例）
        if h >= w:
            new_h = max(1, h // 2)
            new_w = max(1, int(w * (new_h / h)))
        else:
            new_w = max(1, w // 2)
            new_h = max(1, int(h * (new_w / w)))

        # 使用双线性插值下采样
        #img_down = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # 最近邻下采样
        #img_down = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # LANCZOS4下采样
        img_down = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        #img_eq = hist_equalize_float(img_down)
        #img_eq = clahe_equalize_float(img_down)
        #img_eq = MSR(img_down)
        #img_eq= adaptive_contrast_stretch(img_down, low_threshold=0.05, high_threshold=0.95)
        #img_eq = stretch_dark_region(img, dark_ratio=0.3, dark_upper=0.15, min_dynamic=0.2)
        # img_eq = stretch_dark_to_half(img)
        # img_down_eq = stretch_dark_to_half(img_down)

        # img_eq = img
        # img_down_eq = img_down

        # 保存图像
        save_path_down = os.path.join(output_dir, filename)
        save_path_origin = os.path.join(output_dir_origin, filename)
        #save_float32_tiff(save_path_origin, img)
        save_float32_tiff(save_path_down, img_down)
        #cv2.imwrite(save_path, img_down)
        #imageio.imwrite(save_path, img_eq.astype(np.float32)) 

print("所有图像已下采样并保存")
