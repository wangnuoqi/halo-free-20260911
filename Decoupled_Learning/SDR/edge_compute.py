import os
import torch
import numpy as np
import tifffile
import cv2
from torch import nn

def load_tiff_with_log(input_path):
    input = tifffile.imread(input_path)
    input = np.clip(input, 1e-6, None)
    if input is None:
        raise ValueError(f"Failed to read image at {input_path}")

    # 2. 灰度图处理成 [H, W, 1]
    if input.ndim == 2:
        img = np.expand_dims(input, axis=2)
    img = img.astype(np.float32)
    img = np.nan_to_num(img, nan=1e-6, posinf=1e-6, neginf=1e-6)
    img = np.clip(img, 1e-6, None)   
    img = img / (img.max() + 1e-6)
    log_img = np.log(img)
    min_val = log_img.min()
    max_val = 0.0  # 等于 log(1)
    log_norm = (log_img - min_val) / (max_val - min_val + 1e-6)
    #output_path = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/123.tiff"
    #tifffile.imwrite(output_path, log_norm.astype(np.float32), photometric='minisblack')
    #print("Exists after write?", os.path.exists(output_path))
    return log_norm

def save_image(filename, data):
    #img = data.numpy()
    img = np.clip(data,0,255)
    #img = img.transpose(1, 2, 0).astype("uint8")

    cv2.imwrite(filename,img)


class EdgeComputation(nn.Module):
    def __init__(self):
        super(EdgeComputation, self).__init__()

    def forward(self, x):
        # x: (N, C, H, W)
        x_diffx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        x_diffy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

        # 分配一个同样大小的张量，并累加四个方向的差分
        y = torch.zeros_like(x)
        y[:, :, :, 1:]   += x_diffx
        y[:, :, :, :-1]  += x_diffx
        y[:, :, 1:, :]   += x_diffy
        y[:, :, :-1, :]  += x_diffy

        # 通道平均，再除以 4 得到四方向的平均梯度
        y = torch.sum(y, dim=1, keepdim=False)
        y = y / 4.0  # 四个方向的平均
        
        # y_np = y.detach().cpu().numpy().astype(np.float32)
        # output_path = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/123.tiff"
        # tifffile.imwrite(output_path, y_np.astype(np.float32), photometric='minisblack')
        # print("Exists after write?", os.path.exists(output_path))
        return y  # (N, H, W)

if __name__ == "__main__":
    # ------------ 配置 ------------
    input_dir  = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/test/WLS-021544_V4"
    output_dir = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRtest_edge"
    os.makedirs(output_dir, exist_ok=True)

    # ------------ 模型 ------------
    device = torch.device("cpu")
    edge_net = EdgeComputation().to(device)
    edge_net.eval()

    # ------------ 遍历并处理 ------------
    for fname in sorted(os.listdir(input_dir)):
        #if not fname.lower().endswith((".tif", ".tiff")):
        if not fname.lower().endswith((".png")):
            continue

        # 1) 读取图像为 numpy float32
        img_path = os.path.join(input_dir, fname)
        #img = tifffile.imread(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # 如果是灰度图，补一个通道维度
        # if img.ndim == 2:
        #     img = np.expand_dims(img, axis=2)  # H×W → H×W×1
        img = img[:, :, np.newaxis]


        # 2) 转为 (C, H, W) 并放到 batch 维
        img = img.transpose(2, 0, 1)          # H×W×C → C×H×W
        x   = torch.from_numpy(img).unsqueeze(0).to(device)  # 1×C×H×W

        # 3) 前向计算
        with torch.no_grad():
            y = edge_net(x)  # (1, H, W)

        # 4) 保存为 float32 TIFF
        y_np = y.squeeze(0).cpu().numpy()
        out_path = os.path.join(output_dir, fname)
        #tifffile.imwrite(out_path, y_np.astype(np.float32), photometric='minisblack')
        save_image(out_path, y_np)
        print(f"Saved edge map: {out_path}")