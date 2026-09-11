import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import imageio
# a standalone visualizer script that mirrors the edge-mask logic in
#   train_rf_3stage.py
#   Usage example:

#   python /mnt/d/nuoqi/halo-free/Decoupled_Learning/DeepGuidedFilter/metric_cal/
#   lossmask_vis.py \
#     --image /path/to/target.png \
#     --output /path/to/edge_mask.png \
#     --thresh_mode percentile \
#     --thresh 0.4 \
#     --dilate 0 \
#     --normalize_grad


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
    thresh=0.4,
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


def save_mask(mask, out_path):
    mask = mask.squeeze().detach().cpu().numpy()
    mask_uint8 = (mask * 255.0).clip(0, 255).astype(np.uint8)
    imageio.imwrite(out_path, mask_uint8)


def save_grad_vis(grad, out_path):
    grad = grad.squeeze().detach().cpu().numpy()
    if grad.max() > grad.min():
        grad = (grad - grad.min()) / (grad.max() - grad.min())
    grad_uint8 = (grad * 255.0).clip(0, 255).astype(np.uint8)
    imageio.imwrite(out_path, grad_uint8)


def main():
    parser = argparse.ArgumentParser(description="Visualize edge mask used in edge_guided_loss.")
    parser.add_argument("--image", required=True, help="Path to target image (used to build edge mask).")
    parser.add_argument("--output", required=True, help="Output path for edge mask PNG.")
    parser.add_argument("--thresh_mode", default="percentile", choices=["percentile", "fixed"])
    parser.add_argument("--thresh", type=float, default=0.4)
    parser.add_argument("--dilate", type=int, default=0)
    parser.add_argument("--normalize_grad", action="store_true")
    parser.add_argument("--save_grad", default="", help="Optional path to save gradient magnitude visualization.")
    args = parser.parse_args()

    img = load_grayscale(args.image)
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    edge_mask, grad = build_edge_mask(
        tensor,
        thresh_mode=args.thresh_mode,
        thresh=args.thresh,
        dilate=args.dilate,
        normalize_grad=args.normalize_grad,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_mask(edge_mask, args.output)

    if args.save_grad:
        os.makedirs(os.path.dirname(args.save_grad) or ".", exist_ok=True)
        save_grad_vis(grad, args.save_grad)


if __name__ == "__main__":
    main()
