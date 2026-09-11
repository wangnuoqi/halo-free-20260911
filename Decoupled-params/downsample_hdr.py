#!/usr/bin/env python3

import argparse
import os

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Downsample HDR TIFF images by a fixed scale.")
    parser.add_argument("--input-dir", required=True, help="Directory containing input .tiff files.")
    parser.add_argument("--output-dir", required=True, help="Directory to store downsampled .tiff files.")
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Downsampling factor applied to the long edge. Default: 2.",
    )
    return parser.parse_args()


def hist_equalize_float(img):
    img_flat = img.flatten()
    img_min = img_flat.min()
    img_max = img_flat.max()
    img_norm = (img_flat - img_min) / (img_max - img_min)
    hist, bins = np.histogram(img_norm, bins=256, range=[0, 1], density=True)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]
    img_eq = np.interp(img_norm, bins[:-1], cdf)
    img_eq = img_eq.reshape(img.shape)
    return img_eq


def clahe_equalize_float(img):
    img_8bit = (img * 255).clip(0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_8bit)
    img_clahe = img_clahe.astype(np.float32) / 255.0
    return img_clahe


def msr(img, sigmas=(15, 80, 250)):
    img = img.astype(np.float32) + 1e-6
    result = np.zeros_like(img)
    for sigma in sigmas:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        result += np.log(img + 1e-8) - np.log(blur + 1e-8)
    result = result / len(sigmas)
    result = (result - result.min()) / (result.max() - result.min())
    return result


def adaptive_contrast_stretch(img, low_threshold=0.05, high_threshold=0.90):
    img_min = np.percentile(img, low_threshold * 100)
    img_max = np.percentile(img, high_threshold * 100)
    dynamic_range = img_max - img_min

    if dynamic_range < 0.1:
        img_stretch = (img - img_min) / (img_max - img_min + 1e-8)
    else:
        img_stretch = np.clip((img - img_min) / (img_max - img_min + 1e-8), 0, 1)
    return img_stretch


def stretch_dark_to_half(img, mean_thresh=0.3, dark_thresh=0.5):
    img = np.clip(img, 0, 1)
    if img.mean() >= mean_thresh:
        return img

    mask = img < dark_thresh
    dark_pixels = img[mask]
    if dark_pixels.size == 0:
        return img

    dark_norm = (dark_pixels - dark_pixels.min()) / (dark_pixels.max() - dark_pixels.min() + 1e-8)
    hist, bins = np.histogram(dark_norm, bins=256, range=[0, 1], density=True)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]
    dark_eq = np.interp(dark_norm, bins[:-1], cdf)
    stretched = dark_eq * dark_thresh

    img_out = img.copy()
    img_out[mask] = stretched
    return img_out


def save_float32_tiff(save_path, img):
    imageio.imwrite(save_path, img.astype(np.float32))


def downsample_image(img, scale):
    h, w = img.shape[:2]
    if h % 2 != 0:
        h -= 1
    if w % 2 != 0:
        w -= 1
    img = img[:h, :w]

    if h >= w:
        new_h = max(1, h // scale)
        new_w = max(1, int(w * (new_h / h)))
    else:
        new_w = max(1, w // scale)
        new_h = max(1, int(h * (new_w / w)))

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    filenames = sorted(
        name for name in os.listdir(args.input_dir) if name.lower().endswith(".tiff")
    )

    for filename in tqdm(filenames, desc="Processing files", unit="file"):
        file_path = os.path.join(args.input_dir, filename)
        img = imageio.imread(file_path).astype(np.float32)

        if img is None:
            print(f"无法读取: {file_path}")
            continue

        img_down = downsample_image(img, args.scale)
        save_path_down = os.path.join(args.output_dir, filename)
        save_float32_tiff(save_path_down, img_down)

    print(f"所有图像已下采样并保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
