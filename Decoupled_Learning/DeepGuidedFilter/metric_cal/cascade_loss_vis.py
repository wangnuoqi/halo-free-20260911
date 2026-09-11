import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import imageio

# 读取 gt 与 x_coarse0，计算 delta_gt，按 thresh 生成 edge mask，并输出 edge loss 可视化图到
#  halo-free/Decoupled_Learning/DeepGuidedFilter/metric_cal/cascade_edgeloss_vis
#   python /mnt/d/nuoqi/halo-free/Decoupled_Learning/DeepGuidedFilter/metric_cal/cascade_loss_vis.py
#   \
#     --gt /path/to/gt.png \
#     --coarse /path/to/coarse.png \
#     --thresh_mode percentile \
#     --thresh 0.8 \
#     --dilate 0 \
#     --normalize_grad \
#     --save_grad \
#     --prefix demo
#   1. 读入 gt 和 coarse
#   2. 计算残差真值 delta_gt = gt - coarse
#   3. 对 delta_gt 做 Sobel 梯度
#   4. 按固定阈值或分位数阈值得到边缘 mask
#   5. 用这个 mask 截出 delta_gt 中的高梯度区域并保存出来

#   cascade_loss_vis.py 现在会保存：

#   - *_delta_gt.tiff：delta_gt = gt - x_coarse0，float32 原始值，不归一化。
#   - *_delta_grad.tiff：delta_gt 的 Sobel 梯度幅值（float32 原始值）。
#   - *_edge_mask.png：阈值后 edge mask（0/1 映射到 0/255）。
#   - *_grad_loss.png：梯度域 loss 可视化（grad_mag * edge_mask，归一化成 8-bit）。
#   - *_grad.png（可选）：梯度幅值可视化（用于直观看阈值效果）。

def compute_sobel_grad(x, eps=1e-6, normalize=False):
    sobel_x = torch.tensor([[[[-1., 0., 1.],
                              [-2., 0., 2.],
                              [-1., 0., 1.]]]], device=x.device, dtype=x.dtype)
    sobel_y = torch.tensor([[[[-1., -2., -1.],
                              [ 0.,  0.,  0.],
                              [ 1.,  2.,  1.]]]], device=x.device, dtype=x.dtype)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + eps)
    if normalize:
        g = g / (g.amax(dim=(2, 3), keepdim=True) + eps)
    return g, gx, gy


def build_edge_mask(
    target,
    thresh_mode="percentile",
    thresh=0.8,
    dilate=0,
    normalize_grad=True,
):
    g, _, _ = compute_sobel_grad(target, normalize=normalize_grad)

    if thresh_mode == "fixed":
        T = torch.as_tensor(thresh, device=target.device, dtype=target.dtype)
        edge_mask = (g > T).float()
    elif thresh_mode == "percentile":
        q = torch.quantile(g.flatten(1), torch.as_tensor(thresh, device=g.device), dim=1)
        edge_mask = (g > q.view(-1, 1, 1, 1)).float()
    else:
        raise ValueError("thresh_mode must be 'fixed' or 'percentile'")

    if dilate and dilate > 0:
        k = dilate * 2 + 1
        edge_mask = F.max_pool2d(edge_mask, kernel_size=k, stride=1, padding=dilate)

    return edge_mask, g


def load_grayscale(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 2:
        return img.astype("float32")
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray.astype("float32")
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0].astype("float32")
    raise ValueError(f"Unsupported image shape: {img.shape}")


def save_uint8_image(arr, out_path):
    imageio.imwrite(out_path, arr.astype(np.uint8))


def save_float32_tiff(img, save_path):
    """
    直接保存原始 img 为 float32 tiff，不做归一化。
    保留负值和原始动态范围。
    """
    img = img.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32)
    imageio.imwrite(save_path, img)


def save_mask(mask, out_path):
    mask = mask.squeeze().detach().cpu().numpy()
    mask_uint8 = (mask * 255.0).clip(0, 255).astype(np.uint8)
    save_uint8_image(mask_uint8, out_path)


def save_grad_vis(grad, out_path):
    grad = grad.squeeze().detach().cpu().numpy()
    if grad.max() > grad.min():
        grad = (grad - grad.min()) / (grad.max() - grad.min())
    grad_uint8 = (grad * 255.0).clip(0, 255).astype(np.uint8)
    save_uint8_image(grad_uint8, out_path)


def save_loss_vis(loss_map, out_path):
    loss_map = loss_map.squeeze().detach().cpu().numpy()
    if loss_map.max() > loss_map.min():
        loss_map = (loss_map - loss_map.min()) / (loss_map.max() - loss_map.min())
    loss_uint8 = (loss_map * 255.0).clip(0, 255).astype(np.uint8)
    save_uint8_image(loss_uint8, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize edge mask and edge loss map from cascade_loss (delta_gt)."
    )
    parser.add_argument("--gt", required=True, help="Path to GT image.")
    parser.add_argument("--coarse", required=True, help="Path to x_coarse0 image.")
    parser.add_argument(
        "--out_dir",
        default="/mnt/d/nuoqi/halo-free/Decoupled_Learning/DeepGuidedFilter/metric_cal/cascade_edgeloss_vis",
        help="Directory to save visualizations.",
    )
    parser.add_argument("--prefix", default="sample", help="Filename prefix for outputs.")
    parser.add_argument("--thresh_mode", default="percentile", choices=["percentile", "fixed"])
    parser.add_argument("--thresh", type=float, default=0.8)
    parser.add_argument("--dilate", type=int, default=0)
    parser.add_argument("--normalize_grad", action="store_true")
    parser.add_argument("--save_grad", action="store_true", help="Save gradient magnitude visualization.")
    args = parser.parse_args()

    gt = load_grayscale(args.gt)
    coarse = load_grayscale(args.coarse)
    if gt.shape != coarse.shape:
        raise ValueError(f"Shape mismatch: gt {gt.shape} vs coarse {coarse.shape}")

    gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
    coarse_t = torch.from_numpy(coarse).unsqueeze(0).unsqueeze(0)

    delta_gt = gt_t - coarse_t
    edge_mask, grad = build_edge_mask(
        delta_gt,
        thresh_mode=args.thresh_mode,
        thresh=args.thresh,
        dilate=args.dilate,
        normalize_grad=args.normalize_grad,
    )

    # Gradient-domain loss map: grad magnitude masked by edge region.
    grad_mag, gx, gy = compute_sobel_grad(delta_gt, normalize=args.normalize_grad)
    grad_loss_map = grad_mag * edge_mask

    os.makedirs(args.out_dir, exist_ok=True)
    delta_path = os.path.join(args.out_dir, f"{args.prefix}_delta_gt.tiff")
    grad_path_f32 = os.path.join(args.out_dir, f"{args.prefix}_delta_grad.tiff")
    mask_path = os.path.join(args.out_dir, f"{args.prefix}_edge_mask.png")
    loss_path = os.path.join(args.out_dir, f"{args.prefix}_grad_loss.png")

    save_float32_tiff(delta_gt, delta_path)
    save_float32_tiff(grad_mag, grad_path_f32)
    save_mask(edge_mask, mask_path)
    save_loss_vis(grad_loss_map, loss_path)

    if args.save_grad:
        grad_path = os.path.join(args.out_dir, f"{args.prefix}_grad.png")
        save_grad_vis(grad, grad_path)

    print(f"Saved: {delta_path}")
    print(f"Saved: {grad_path_f32}")
    print(f"Saved: {mask_path}")
    print(f"Saved: {loss_path}")
    if args.save_grad:
        print(f"Saved: {grad_path}")


if __name__ == "__main__":
    main()
