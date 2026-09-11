import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import imageio

# output edge from otus method
# ====== 你只需要改这里 ======
INPUT_DIR = "/mnt/d/nuoqi/S2R-HDR/data/test_vivo/testlabel"
# ===========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "otsu_edge_gt_vis")
OUTPUT_MASK_DIR = os.path.join(OUTPUT_ROOT, "mask")
OUTPUT_GRAD_DIR = os.path.join(OUTPUT_ROOT, "grad")
OUTPUT_OVERLAY_DIR = os.path.join(OUTPUT_ROOT, "overlay")


def compute_sobel_grad(x, eps=1e-6, normalize=False):
    sobel_x = torch.tensor([[[[-1., 0., 1.],
                              [-2., 0., 2.],
                              [-1., 0., 1.]]]], device=x.device, dtype=x.dtype)
    sobel_y = torch.tensor([[[[-1., -2., -1.],
                              [0., 0., 0.],
                              [1., 2., 1.]]]], device=x.device, dtype=x.dtype)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + eps)
    if normalize:
        g = g / (g.amax(dim=(2, 3), keepdim=True) + eps)
    return g


def batched_otsu_threshold(g, num_bins=256, eps=1e-6, min_delta=1e-6):
    assert g.dim() == 4 and g.shape[1] == 1, "g must be (B,1,H,W)"
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")

    B = g.shape[0]
    flat = g.flatten(1)

    g_min = flat.min(dim=1).values
    g_max = flat.max(dim=1).values
    g_span = g_max - g_min
    span = g_span.clamp_min(eps)

    norm = ((flat - g_min[:, None]) / span[:, None]).clamp(0.0, 1.0)
    bin_idx = (norm * (num_bins - 1)).long()

    offsets = (torch.arange(B, device=g.device) * num_bins)[:, None]
    hist_idx = (bin_idx + offsets).reshape(-1)
    hist = torch.bincount(hist_idx, minlength=B * num_bins).reshape(B, num_bins).to(g.dtype)

    prob = hist / hist.sum(dim=1, keepdim=True).clamp_min(eps)
    centers = torch.linspace(0.0, 1.0, steps=num_bins, device=g.device, dtype=g.dtype).view(1, -1)

    omega = torch.cumsum(prob, dim=1)
    mu = torch.cumsum(prob * centers, dim=1)
    mu_total = mu[:, -1:]
    denom = (omega * (1.0 - omega)).clamp_min(eps)
    sigma_between = ((mu_total * omega - mu) ** 2) / denom
    sigma_between[:, 0] = 0.0
    sigma_between[:, -1] = 0.0

    best_bin = torch.argmax(sigma_between, dim=1)
    t_norm = centers[0, best_bin]
    thresh = g_min + t_norm * span

    valid_span = g_span > min_delta
    high = g_max - min_delta
    thresh = torch.where(valid_span, torch.minimum(thresh, high), g_max)
    return thresh


def build_otsu_edge_mask(gray_np, normalize_grad=True, eps=1e-6):
    tensor = torch.from_numpy(gray_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    g = compute_sobel_grad(tensor, eps=eps, normalize=normalize_grad)

    T = batched_otsu_threshold(g, num_bins=256, eps=eps, min_delta=eps)
    edge_mask = (g > T.view(-1, 1, 1, 1)).float()

    edge_count = edge_mask.flatten(1).sum(dim=1)
    no_edge = edge_count <= 0
    if no_edge.any():
        flat_g = g.flatten(1)
        max_idx = flat_g.argmax(dim=1, keepdim=True)
        fallback = torch.zeros_like(flat_g)
        fallback.scatter_(1, max_idx, 1.0)
        flat_mask = edge_mask.flatten(1)
        flat_mask[no_edge] = fallback[no_edge]
        edge_mask = flat_mask.view_as(edge_mask)

    return edge_mask.squeeze(0).squeeze(0), g.squeeze(0).squeeze(0), float(T.item())


def load_grayscale(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if img.ndim == 2:
        gray = img.astype(np.float32)
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    elif img.ndim == 3 and img.shape[2] == 1:
        gray = img[:, :, 0].astype(np.float32)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    return gray


def to_uint8_norm(x):
    x = x.astype(np.float32)
    if x.max() > x.min():
        x = (x - x.min()) / (x.max() - x.min())
    else:
        x = np.zeros_like(x, dtype=np.float32)
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def save_float32_tiff(img, save_path):
    img = img.astype(np.float32)
    imageio.imwrite(save_path, img)


def build_overlay(gray_np, edge_mask_np):
    base = to_uint8_norm(gray_np)
    color = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    overlay = color.copy()
    overlay[edge_mask_np > 0] = np.array([0, 0, 255], dtype=np.uint8)
    merged = cv2.addWeighted(color, 0.75, overlay, 0.25, 0)
    return merged


def ensure_dirs():
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_GRAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_OVERLAY_DIR, exist_ok=True)


def collect_images(input_dir):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and os.path.splitext(name.lower())[1] in exts:
            files.append(path)
    return files


def main():
    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"INPUT_DIR not found: {INPUT_DIR}")

    ensure_dirs()
    img_paths = collect_images(INPUT_DIR)
    if not img_paths:
        raise RuntimeError(f"No images found in: {INPUT_DIR}")

    print(f"[Info] Found {len(img_paths)} images.")
    print(f"[Info] Output root: {OUTPUT_ROOT}")

    for idx, img_path in enumerate(img_paths, start=1):
        name = os.path.splitext(os.path.basename(img_path))[0]
        gray = load_grayscale(img_path)

        edge_mask_t, grad_t, threshold = build_otsu_edge_mask(gray, normalize_grad=True)
        edge_mask = edge_mask_t.detach().cpu().numpy().astype(np.uint8)
        grad = grad_t.detach().cpu().numpy()

        mask_u8 = (edge_mask * 255).astype(np.uint8)
        grad_u8 = to_uint8_norm(grad)
        overlay = build_overlay(gray, edge_mask)

        mask_path = os.path.join(OUTPUT_MASK_DIR, f"{name}_mask.png")
        grad_path = os.path.join(OUTPUT_GRAD_DIR, f"{name}_grad.png")
        grad_f32_path = os.path.join(OUTPUT_GRAD_DIR, f"{name}_grad_f32.tiff")
        overlay_path = os.path.join(OUTPUT_OVERLAY_DIR, f"{name}_overlay.png")

        imageio.imwrite(mask_path, mask_u8)
        imageio.imwrite(grad_path, grad_u8)
        save_float32_tiff(grad, grad_f32_path)
        imageio.imwrite(overlay_path, overlay)

        print(f"[{idx}/{len(img_paths)}] {os.path.basename(img_path)} | otsu_T={threshold:.6f}")

    print("[Done] Otsu edge visualization saved.")


if __name__ == "__main__":
    main()
