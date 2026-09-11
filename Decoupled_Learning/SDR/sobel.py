import os
import cv2
import numpy as np

def sobel_edge_map(image: np.ndarray, thresh: float = 0.1) -> np.ndarray:
    """
    对 float32 [0,1] 的图像做 Sobel 边缘检测并二值化。
    返回 uint8 的二值图，255 表示边缘。
    """
    # 1) 转灰度
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)

    # 2) 一阶 Sobel
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # 3) 幅值 & 归一化
    mag = cv2.magnitude(gx, gy)
    if mag.max() > 0:
        mag /= mag.max()

    # 4) 二值化阈值
    edge = (mag > thresh).astype(np.uint8) * 255
    return edge

def batch_process_tiff(input_folder: str,
                       output_folder: str,
                       thresh: float = 0.05):
    """
    循环读取 input_folder 下所有 .tif/.tiff 文件，
    对每张图做 Sobel 边缘检测，并保存到 output_folder
    """
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(('.tif', '.tiff')):
            continue
        in_path = os.path.join(input_folder, fname)

        # 读取 TIFF 图（保持原始深度）
        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] 读取失败: {in_path}")
            continue

        # 转成 float32 [0,1]
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0

        # 计算边缘
        edges = sobel_edge_map(img, thresh)

        # 构造输出路径
        base, _ = os.path.splitext(fname)
        out_path = os.path.join(output_folder, f"{base}_edge.png")

        # 保存
        cv2.imwrite(out_path, edges)
        print(f"Saved edge map: {out_path}")

# —— 调用示例 —— #
if __name__ == "__main__":
    input_dir  = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR512_L0L1_vallabel_0.3"
    output_dir = "/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR512_l0l1_label_0.3_edge"
    batch_process_tiff(input_dir, output_dir, thresh=0.1)
