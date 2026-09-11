import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import imageio

# DeltaGT Otsu edge visualization:
# delta_gt = gt - coarse
GT_DIR = "/mnt/d/nuoqi/S2R-HDR/data/test_vivo/testlabel"
COARSE_DIR = "/mnt/d/nuoqi/S2R-HDR/data/test_vivo/df_output"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "otsu_edge_delta_vis")
OUTPUT_DELTA_DIR = os.path.join(OUTPUT_ROOT, "delta")
OUTPUT_MASK_DIR = os.path.join(OUTPUT_ROOT, "mask")
OUTPUT_GRAD_DIR = os.path.join(OUTPUT_ROOT, "grad")
OUTPUT_OVERLAY_DIR = os.path.join(OUTPUT_ROOT, "overlay")


def compute_sobel_grad(x, eps=1e-6, normalize=False):
    sobel_x = torch.tensor([[[[-1.0, 0.0, 1.0],
                              [-2.0, 0.0, 2.0],
                              [-1.0, 0.0, 1.0]]]], device=x.device, dtype=x.dtype)
    sobel_y = torch.tensor([[[[-1.0, -2.0, -1.0],
                              [0.0, 0.0, 0.0],
                              [1.0, 2.0, 1.0]]]], device=x.device, dtype=x.dtype)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    g = torch.sqrt(gx * gx + gy * gy + eps)
    if normalize:
        g = g / (g.amax(dim=(2, 3), keepdim=True) + eps)
    return g, gx, gy


def batched_otsu_threshold(g, num_bins=256, eps=1e-6, min_delta=1e-6):
    assert g.dim() == 4 and g.shape[1] == 1, "g must be (B,1,H,W)"
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")

    batch_size = g.shape[0]
    flat = g.flatten(1)

    g_min = flat.min(dim=1).values
    g_max = flat.max(dim=1).values
    g_span = g_max - g_min
    span = g_span.clamp_min(eps)

    norm = ((flat - g_min[:, None]) / span[:, None]).clamp(0.0, 1.0)
    bin_idx = (norm * (num_bins - 1)).long()

    offsets = (torch.arange(batch_size, device=g.device) * num_bins)[:, None]
    hist_idx = (bin_idx + offsets).reshape(-1)
    hist = torch.bincount(hist_idx, minlength=batch_size * num_bins).reshape(batch_size, num_bins).to(g.dtype)

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


def build_otsu_edge_mask_from_delta(delta_np, normalize_grad=True, eps=1e-6):
    tensor = torch.from_numpy(delta_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    g, _, _ = compute_sobel_grad(tensor, eps=eps, normalize=normalize_grad)

    threshold = batched_otsu_threshold(g, num_bins=256, eps=eps, min_delta=eps)
    edge_mask = (g > threshold.view(-1, 1, 1, 1)).float()

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

    return edge_mask.squeeze(0).squeeze(0), g.squeeze(0).squeeze(0), float(threshold.item())


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


def crop_pair_to_common_shape(a, b):
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    return a[:height, :width], b[:height, :width]


def to_uint8_norm(x):
    x = x.astype(np.float32)
    if x.max() > x.min():
        x = (x - x.min()) / (x.max() - x.min())
    else:
        x = np.zeros_like(x, dtype=np.float32)
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def save_float32_tiff(img, save_path):
    imageio.imwrite(save_path, img.astype(np.float32))


def build_overlay(delta_np, edge_mask_np):
    base = to_uint8_norm(delta_np)
    color = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    overlay = color.copy()
    overlay[edge_mask_np > 0] = np.array([0, 0, 255], dtype=np.uint8)
    return cv2.addWeighted(color, 0.75, overlay, 0.25, 0)


def ensure_dirs():
    os.makedirs(OUTPUT_DELTA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_GRAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_OVERLAY_DIR, exist_ok=True)


def collect_images(input_dir):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = {}
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and os.path.splitext(name.lower())[1] in exts:
            stem = os.path.splitext(name)[0]
            files[stem] = path
    return files


def normalize_pair_key(stem):
    if stem.endswith("_pred"):
        return stem[:-5]
    return stem


def main():
    if not os.path.isdir(GT_DIR):
        raise FileNotFoundError(f"GT_DIR not found: {GT_DIR}")
    if not os.path.isdir(COARSE_DIR):
        raise FileNotFoundError(f"COARSE_DIR not found: {COARSE_DIR}")

    ensure_dirs()

    gt_files = collect_images(GT_DIR)
    coarse_files = collect_images(COARSE_DIR)
    gt_map = {normalize_pair_key(name): path for name, path in gt_files.items()}
    coarse_map = {normalize_pair_key(name): path for name, path in coarse_files.items()}
    common_names = sorted(set(gt_map) & set(coarse_map))

    if not common_names:
        raise RuntimeError("No matched filenames found between GT_DIR and COARSE_DIR")

    print(f"[Info] Matched {len(common_names)} image pairs.")
    print(f"[Info] Output root: {OUTPUT_ROOT}")

    for idx, name in enumerate(common_names, start=1):
        gt = load_grayscale(gt_map[name])
        coarse = load_grayscale(coarse_map[name])
        gt, coarse = crop_pair_to_common_shape(gt, coarse)

        delta_gt = gt - coarse
        edge_mask_t, grad_t, threshold = build_otsu_edge_mask_from_delta(delta_gt, normalize_grad=True)

        edge_mask = edge_mask_t.detach().cpu().numpy().astype(np.uint8)
        grad = grad_t.detach().cpu().numpy().astype(np.float32)

        delta_u8 = to_uint8_norm(delta_gt)
        mask_u8 = (edge_mask * 255).astype(np.uint8)
        grad_u8 = to_uint8_norm(grad)
        overlay = build_overlay(delta_gt, edge_mask)

        delta_path = os.path.join(OUTPUT_DELTA_DIR, f"{name}_delta.png")
        delta_f32_path = os.path.join(OUTPUT_DELTA_DIR, f"{name}_delta_f32.tiff")
        mask_path = os.path.join(OUTPUT_MASK_DIR, f"{name}_mask.png")
        grad_path = os.path.join(OUTPUT_GRAD_DIR, f"{name}_grad.png")
        grad_f32_path = os.path.join(OUTPUT_GRAD_DIR, f"{name}_grad_f32.tiff")
        overlay_path = os.path.join(OUTPUT_OVERLAY_DIR, f"{name}_overlay.png")

        imageio.imwrite(delta_path, delta_u8)
        save_float32_tiff(delta_gt, delta_f32_path)
        imageio.imwrite(mask_path, mask_u8)
        imageio.imwrite(grad_path, grad_u8)
        save_float32_tiff(grad, grad_f32_path)
        imageio.imwrite(overlay_path, overlay)

        print(
            f"[{idx}/{len(common_names)}] {name} | "
            f"otsu_T={threshold:.6f} | shape={delta_gt.shape[0]}x{delta_gt.shape[1]}"
        )

    print("[Done] DeltaGT Otsu edge visualization saved.")


if __name__ == "__main__":
    main()
