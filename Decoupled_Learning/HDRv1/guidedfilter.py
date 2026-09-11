from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 1) BoxFilter: 输出的是窗口内“求和”（不做平均）
# -----------------------------
class BoxFilter(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        self.r = int(r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W]
        r = self.r
        k = 2 * r + 1
        n, c, h, w = x.shape
        weight = x.new_ones((c, 1, k, k))  # sum kernel
        return F.conv2d(x, weight, padding=r, groups=c)


# -----------------------------
# 2) GuidedFilter
# -----------------------------
class GuidedFilter(nn.Module):
    def __init__(self, r: int, eps: float = 1e-3):
        super().__init__()
        self.r = int(r)
        self.eps = float(eps)
        self.boxfilter = BoxFilter(r)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: guidance, [N, 1 or C, H, W]
        y: input,    [N, C,      H, W]
        """
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert (c_x == 1) or (c_x == c_y)
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        ones = torch.ones((n_x, 1, h_x, w_x), device=x.device, dtype=x.dtype)
        N = self.boxfilter(ones)  # [N,1,H,W]，边界处会比 k*k 小

        mean_x = self.boxfilter(x) / N
        mean_y = self.boxfilter(y) / N

        # 如果 x 是 1 通道、y 是多通道：把 x 广播到 y 的通道数
        if c_x == 1 and c_y > 1:
            x_for_ops = x.expand(-1, c_y, -1, -1)
            mean_x = mean_x.expand(-1, c_y, -1, -1)
        else:
            x_for_ops = x

        cov_xy = self.boxfilter(x_for_ops * y) / N - mean_x * mean_y
        var_x  = self.boxfilter(x_for_ops * x_for_ops) / N - mean_x * mean_x

        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x

        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        out = mean_A * x_for_ops + mean_b
        return out


# -----------------------------
# 3) 读写图片（尽量用 cv2；没有就回退 PIL）
# -----------------------------
def read_image(path: Path):
    """
    返回: img (H,W,C) float32, range尽量是[0,1]（若是HDR/EXR可能>1）
    """
    suffix = path.suffix.lower()
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"cv2.imread 读取失败: {path}")

        # 通道处理：cv2 读入是 BGR
        if img.ndim == 2:
            img = img[..., None]
        else:
            img = img[..., ::-1]  # BGR->RGB

        # dtype 归一化
        if img.dtype == "uint8":
            img = img.astype("float32") / 255.0
        elif img.dtype == "uint16":
            img = img.astype("float32") / 65535.0
        else:
            img = img.astype("float32")  # float/其他45455

        return img
    except ImportError:
        # fallback PIL
        from PIL import Image
        im = Image.open(path)
        im = im.convert("RGB") if im.mode != "RGB" else im
        img = (torch.from_numpy(__import__("numpy").array(im)).float() / 255.0).numpy()
        return img.astype("float32")


def save_tiff_float(path: Path, img_float):
    """
    保存 float32 TIFF：
    - 不 clip、不归一化：保留真实数值范围（适合 HDR/中间结果）
    - 支持 (H,W)、(H,W,1)、(H,W,3)、(H,W,4)
    """
    import numpy as np
    import tifffile as tiff

    x = np.asarray(img_float)

    # (H,W,1) -> (H,W)
    if x.ndim == 3 and x.shape[2] == 1:
        x = x[:, :, 0]

    x = x.astype(np.float32)

    # tifffile 对 (H,W) 和 (H,W,C) 都支持
    tiff.imwrite(str(path), x)


def rgb_to_gray(img_rgb):
    """
    img_rgb: (H,W,3) float32
    """
    # 简单的 Rec.601 luma
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray[..., None]


# -----------------------------
# 4) 主流程：遍历 input_dir -> guided filter -> out_dir
# -----------------------------
def main():
    input_dir = Path("/mnt/d/nuoqi/S2R-HDR/data/pipelinev1/input_test_log")
    out_dir = Path("/mnt/d/nuoqi/S2R-HDR/data/mixed_lavel_singlehdr/result/guidedoutput")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 你可以调这些参数
    r =  3         # 半径，越大越平滑
    eps =  0.2     # 正则项，越大越“保守”
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gf = GuidedFilter(r=r, eps=eps).to(device).eval()

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".hdr", ".exr"}
    paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]

    if not paths:
        print(f"在 {input_dir} 没找到图片（支持扩展名：{sorted(exts)}）")
        return

    print(f"找到 {len(paths)} 张图片，device={device}，输出到：{out_dir}")

    for p in paths:
        img = read_image(p)  # (H,W,C)
        h, w, c = img.shape

        # 太小会触发 guided filter 的 assert（窗口撑不开）
        if h <= 2 * r + 1 or w <= 2 * r + 1:
            print(f"[跳过] 图片太小：{p.name} shape={img.shape}")
            continue

        # x: 引导图（灰度）
        if c >= 3:
            x = rgb_to_gray(img[..., :3])  # (H,W,1)
            y = img[..., :3]               # (H,W,3)
        else:
            # 单通道：x=y
            x = img[..., :1]
            y = img[..., :1]

        # -> torch: [1,C,H,W]
        x_t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
        y_t = torch.from_numpy(y).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            out_t = gf(x_t, y_t)

        out = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # 这里为了方便展示，统一存 PNG（会把结果 clamp 到 [0,1]）
        out_name = p.stem + "_guided.tiff"
        save_tiff_float(out_dir / out_name, out)

        print(f"[OK] {p.name} -> {out_name}")

    print("全部完成。")


if __name__ == "__main__":
    main()
