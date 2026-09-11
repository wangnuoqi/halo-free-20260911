import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

#### 边缘检测结果可视化(Sobel) ####

# 你的Sobel梯度函数
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

# === 1. 读入图片 ===
img_path = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRV1/2012_003969-input_pred.png'         # 修改成你的图片路径
img = Image.open(img_path)   # 转成灰度图
img_np = np.array(img, dtype=np.float32) / 255.0   # 归一化到[0,1]

# === 2. 转为PyTorch格式 ===
img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
img_tensor = img_tensor  # 如果有GPU，可用 .cuda()

# === 3. 计算Sobel边缘 ===
with torch.no_grad():
    grad_tensor = compute_sobel_grad(img_tensor)

# === 4. 转成图片保存 ===
grad_np = grad_tensor.squeeze().cpu().numpy()
grad_np = np.clip(grad_np / grad_np.max(), 0, 1) * 255  # 归一化到0-255
grad_img = Image.fromarray(grad_np.astype(np.uint8))
grad_img.save('/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDRV1/sobel_output.png')

print("已保存边缘检测结果到 sobel_output.png")
