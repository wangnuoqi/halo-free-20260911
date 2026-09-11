import os
import cv2
from tqdm import tqdm
import numpy as np

# 输入输出路径
#input_dir = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRinput"
input_dir_test = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/input_V4' 

output_dir = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/down_sample_2_lanczos_train"
cleaned_dir = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/halo_free_test'
# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

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

# 遍历所有 PNG 文件
for filename in tqdm(os.listdir(input_dir_test), desc="Processing files", unit="file"):
    if filename.lower().endswith(".png"):
        file_path = os.path.join(input_dir_test, filename)

        # 读取图像
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img_cleaned = remove_halo_2d(img)
        save_path_cleaned = os.path.join(cleaned_dir, filename)
        cv2.imwrite(save_path_cleaned, img_cleaned)

        if img is None:
            print(f"无法读取: {file_path}")
            continue

        # 获取原始尺寸
        h, w = img.shape[:2]
        if h % 2 != 0:
            h -= 1
        if w % 2 != 0:
            w -= 1
        img = img[:h, :w]  # 裁剪图像

        # 计算目标尺寸（长边缩小 8 倍，保持比例）
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

        # 保存图像
        save_path = os.path.join(output_dir, filename)
        #cv2.imwrite(save_path, img_down)

print("所有图像已下采样并保存到:", output_dir)
