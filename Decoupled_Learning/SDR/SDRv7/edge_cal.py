import os
import numpy as np
import cv2

# ----------- 两个边缘函数（直接粘贴上面代码即可） ------------
############# 计算上下左右梯度 ###############
def edgecal_vertical(x):
    x = np.asarray(x)
    orig_shape = x.shape
    # 保证至少4维
    if x.ndim == 2:
        x = x[None, None, ...]    # [1,1,H,W]
    elif x.ndim == 3:
        if x.shape[0] == 3 or x.shape[0] == 1:
            x = x[None, ...]
        elif x.shape[2] == 3 or x.shape[2] == 1:
            x = np.transpose(x, (2,0,1))[None]
        else:
            x = x[None, ...]
    B, C, H, W = x.shape
    y = np.zeros_like(x)
    x_diffx = np.abs(x[..., 1:] - x[..., :-1])
    x_diffy = np.abs(x[..., 1:, :] - x[..., :-1, :])
    y[..., 1:] += x_diffx
    y[..., :-1] += x_diffx
    y[..., 1:, :] += x_diffy
    y[..., :-1, :] += x_diffy
    if C == 1:
        y_out = y[:, 0:1] / 4
    else:
        y_out = np.sum(y, axis=1, keepdims=True) / (3*4)
    # 返回shape兼容
    if orig_shape == (H, W):
        return y_out[0, 0]
    elif len(orig_shape) == 3 and (orig_shape[0] == 1 or orig_shape[0] == 3):
        return y_out[0]
    elif len(orig_shape) == 3 and orig_shape[2] == 1:
        return y_out[0, 0]
    elif len(orig_shape) == 4:
        return y_out
    else:
        return y_out[0]

########## 周边正方形周边差计算梯度 根据radius计算不同的结果 ########
# def edgecal_1(guidance, radius=1):
#     x = np.asarray(guidance)
#     if x.ndim == 2:
#         x = x[..., None]
#     if x.ndim == 3:
#         x = x[None]
#     B, H, W, C = x.shape
#     edge_maps = []
#     pad = radius
#     for b in range(B):
#         guidance_i = x[b]
#         edge_map = np.zeros((H, W), dtype=np.float32)
#         guide_pad = np.pad(guidance_i, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
#         for dy in range(-radius, radius+1):
#             for dx in range(-radius, radius+1):
#                 if dy == 0 and dx == 0:
#                     continue
#                 crop = guide_pad[pad+dy:pad+dy+H, pad+dx:pad+dx+W, :]
#                 diff = np.sum(guidance_i - crop, axis=2)
#                 edge_map += np.abs(diff)
#         edge_maps.append(edge_map)
#     edge_maps = np.stack(edge_maps, axis=0)
#     if edge_maps.shape[0] == 1:
#         return edge_maps[0]
#     else:
#         return edge_maps

def edgecal_surround(x, radius=2):
    """
    输入: x [H, W]，单通道灰度图
    输出: edge_map [H, W]，每像素与8邻域的平均绝对差分
    """
    if x.ndim != 2:
        raise ValueError("Input must be a 2D array!")

    H, W = x.shape
    x_pad = np.pad(x, ((radius, radius), (radius, radius)), mode='reflect')

    edge_map = np.zeros((H, W), dtype=np.float32)
    neighbor_count = 0
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if dy == 0 and dx == 0:
                continue   # 跳过自身
            neighbor = x_pad[radius+dy:H+radius+dy, radius+dx:W+radius+dx]
            diff = np.abs(x - neighbor)
            edge_map += diff
            neighbor_count += 1

    edge_map /= neighbor_count  # 归一化
    return edge_map
    

###### 采用方差判断边缘 (效果不佳) ########
def edgecal_variance(img, radius=1):
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 2:
        x = x[..., None]   # 灰度图 -> [H,W,1]
    H, W, C = x.shape
    pad = radius
    x_pad = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    edge_map = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            window = x_pad[i:i+2*radius+1, j:j+2*radius+1, :]  # 取窗口 [win, win, C]
            vals = window.reshape(-1, C)                       # 拉平成 [win*win, C]
            local_var = np.var(vals, axis=0)                   # 每通道分别计算方差，shape=(C,)
            edge_map[i, j] = local_var.mean()                  # 对所有通道方差做均值（也可取max等）
    return edge_map 

###### 周边四像素计算边缘信息(同上) #######
from scipy.ndimage import uniform_filter
import torch
import torch.nn.functional as F
'''
def edgecal_4(x, radius=2):
    if x.ndim != 2:
        raise ValueError("Input must be a 2D array!")
    H, W = x.shape
    x_pad = np.pad(x, ((radius, radius), (radius, radius)), mode='reflect')
    edge_map = np.zeros((H, W), dtype=np.float32)
    neighbor_count = 0
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if dy == 0 and dx == 0:
                continue
            neighbor = x_pad[radius+dy:H+radius+dy, radius+dx:W+radius+dx]
            diff = np.abs(x - neighbor)
            edge_map += diff
            neighbor_count += 1
    edge_map /= neighbor_count
    return edge_map

def salient_edge_map(x, radius=2, radius_var=4, var_thresh=None, thresh_factor=2.0):
    edge_map = edgecal_4(x, radius=radius)
    edge_map = edge_map.astype(np.float32) / 255.0
    # 使用滑动窗口估计局部方差
    mean = uniform_filter(edge_map, size=radius_var*2+1)
    mean_sq = uniform_filter(edge_map**2, size=radius_var*2+1)
    var_map = mean_sq - mean**2
    print(var_map.max(), var_map.mean())
    if var_thresh is None:
        var_thresh = var_map.mean() * thresh_factor
    # 提取方差大于阈值的区域为显著边缘
    salient_edges = np.zeros_like(edge_map)
    salient_edges[var_map > var_thresh] = edge_map[var_map > var_thresh]
    print(salient_edges.max(), salient_edges.mean())
    return salient_edges
'''
def edgecal_4_tensor(x, radius=2):
    """
    x: [1, 1, H, W] float32 tensor, 已归一化到 [0,1]，在 GPU 上
    输出: edge_map 同尺寸 [1, 1, H, W]
    """
    B, C, H, W = x.shape
    edge_map = torch.zeros_like(x)
    neighbor_count = 0

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = F.pad(x, (radius, radius, radius, radius), mode='reflect')
            shifted = shifted[:, :, radius + dy:H + radius + dy, radius + dx:W + radius + dx]
            diff = torch.abs(x - shifted)
            edge_map += diff
            neighbor_count += 1

    edge_map /= neighbor_count
    return edge_map
######## 只计算保留的大边缘 ########
def salient_edge_map_tensor(x, radius=2, radius_var=4, var_thresh=None, soft=True, soft_power=10.0, thresh_factor=2.0):
    """
    x: [1, 1, H, W] float32 tensor, 已归一化到 [0,1]，在 GPU 上
    """
    edge_map = edgecal_4_tensor(x, radius=radius)

    # 计算局部方差
    k = radius_var * 2 + 1
    mean = F.avg_pool2d(edge_map, k, stride=1, padding=radius_var)
    mean_sq = F.avg_pool2d(edge_map ** 2, k, stride=1, padding=radius_var)
    var_map = mean_sq - mean ** 2

    if var_thresh is None:
        var_thresh = var_map.mean() * thresh_factor

    # 显著边缘提取
    salient_edges = torch.zeros_like(edge_map)
    if soft:
        delta = var_thresh * 0.5

        # 线性 mask：中心区为 1，两侧线性衰减到 0
        mask = (var_map - (var_thresh - delta)) / (2 * delta)
        mask = torch.clamp(mask, 0.0, 1.0)
        salient_edges = edge_map * mask
    else:
        # 原始硬过滤方式
        salient_edges = torch.zeros_like(edge_map)
        salient_edges[var_map > var_thresh] = edge_map[var_map > var_thresh]

    print(f"var_map.max: {var_map.max().item():.6f}, mean: {var_map.mean().item():.6f}")
    print(f"salient_edges.max: {salient_edges.max().item():.6f}, mean: {salient_edges.mean().item():.6f}")
    return salient_edges

######## 重新写了一个按照原文实现的边缘检测函数 ######
def edge_computation_origin(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    计算邻域差分边缘图 (黑色边缘，白色非边缘)
    输入: image (H, W) 或 (H, W, C)
    输出: edge_map (H, W) uint8 (0-255)
    """
    # 转 float 并缩放
    img = image.astype(np.float32) / scale

    # 转灰度 (∑_c)
    if img.ndim == 3:
        img_gray = img.sum(axis=2)  # (H,W)
    else:
        img_gray = img

    H, W = img_gray.shape
    radius = 2
    edge_map = np.zeros((H, W), dtype=np.float32)

    # 遍历每个像素
    for y in range(H):
        for x in range(W):
            center_val = img_gray[y, x]
            total_diff = 0.0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    if ny < 0 or ny >= H or nx < 0 or nx >= W:
                        continue
                    if dy == 0 and dx == 0:
                        continue
                    neighbor_val = img_gray[ny, nx]
                    total_diff += abs(center_val - neighbor_val)

            edge_map[y, x] = total_diff

    # 归一化到 [0,255]
    edge_map -= edge_map.min()
    if edge_map.max() > 0:
        edge_map /= edge_map.max()
    edge_map = (edge_map * 255).astype(np.uint8)

    # 反色处理 (边缘黑，背景白)
    edge_map = 255 - edge_map

    return edge_map

# ----------- 主体脚本 ------------

#input_dir = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRinput_label'
#input_dir = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/WLS-021544_V4'
input_dir = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRinput'
#input_dir = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/input_V4'

#output_dir_variance = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_edge_V7_test_label_variance'
output_dir_variance = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/SDR_v7/guided_edge_train_input'
#output_dir_main = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_edge_V7_test_label_main'
#output_dir_surround = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_edge_V7_test_label_surround'
#output_dir_vertical = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_edge_V7_test_label_vertical'
#os.makedirs(output_dir, exist_ok=True)

file_list = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
print(f'共找到{len(file_list)}个PNG文件')

for fname in file_list:
    img_path = os.path.join(input_dir, fname)
    # 以原始数据类型读取，兼容灰度和RGB
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f'跳过无法读取: {fname}')
        continue
    # 灰度图 shape [H,W]，RGB shape [H,W,3]
    img_float = img.astype(np.float32)

    # 边缘法1
    #edge_vertical = edgecal_vertical(img_float)
    # 边缘法2
    #edge_variance = edgecal_variance(img_float, radius=1)
    # 边缘法3
    #edge_surround = edgecal_surround(img_float, radius=2)
    # 边缘法5
    edge_origin = edge_computation_origin(img_float)
    
    #边缘法4
    # img_float = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to('cuda')
    # edge4 = salient_edge_map_tensor(img_float, radius=2, radius_var=4, var_thresh=None,soft=True, soft_power=10.0, thresh_factor=3.0)
    # edge4 = edge4.squeeze().cpu().numpy()
    # 归一化到0~255便于可视化
    def norm255(arr):
        arr = arr.astype(np.float32)
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
        else:
            arr = np.zeros_like(arr)
        return np.round(arr).astype(np.uint8)

    edge_origin_out = norm255(edge_origin)
    #edge_vertical_out = norm255(edge_vertical)
    #edge_surround_out = norm255(edge_surround)
    #edge_variance_out = norm255(edge_variance)
    #edge4_out = norm255(edge4 )

    base = os.path.splitext(fname)[0]
    save1 = os.path.join(output_dir_variance, f"{base}_edge.png")
    #save2 = os.path.join(output_dir_surround, f"{base}_edge.png")
    #save3 = os.path.join(output_dir_vertical, f"{base}_edge.png")
    #save = os.path.join(output_dir, f"{base}_edge.png")
    edge_origin_out = edge_origin_out.squeeze()
    cv2.imwrite(save1, edge_origin_out)
    #cv2.imwrite(save2, edge_surround_out)
    #cv2.imwrite(save3, edge_variance_out)

    print(f"保存好了狗熊")

print("全部处理完成！恭喜你，狗熊！")
