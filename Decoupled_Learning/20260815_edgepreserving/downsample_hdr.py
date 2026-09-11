import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from tqdm import tqdm


INPUT_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/test_fei/testinput")
OUTPUT_DIR = Path("/mnt/d/nuoqi/S2R-HDR/data/test_fei/testinput_downsample")


def read_float32_tiff(path):
    return iio.imread(path).astype(np.float32)


def save_float32_tiff(path, img):
    iio.imwrite(path, img.astype(np.float32))


def crop_to_even(img):
    h, w = img.shape[:2]
    if h % 2 != 0:
        h -= 1
    if w % 2 != 0:
        w -= 1
    return img[:h, :w]


def compute_half_long_edge_size(height, width):
    if height >= width:
        new_height = max(1, height // 2)
        new_width = max(1, int(width * (new_height / height)))
    else:
        new_width = max(1, width // 2)
        new_height = max(1, int(height * (new_width / width)))
    return new_height, new_width


def resize_lanczos_float32(img, new_height, new_width):
    pil_img = Image.fromarray(img.astype(np.float32), mode="F")
    resized = pil_img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32)


def downsample_half_long_edge(img):
    img_even = crop_to_even(img)
    new_height, new_width = compute_half_long_edge_size(*img_even.shape[:2])
    return resize_lanczos_float32(img_even, new_height, new_width)


def iter_tiff_files(input_dir):
    for name in sorted(os.listdir(input_dir)):
        if name.lower().endswith(".tiff"):
            yield name


def process_directory(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in tqdm(list(iter_tiff_files(input_dir)), desc="Processing files", unit="file"):
        input_path = input_dir / filename
        output_path = output_dir / filename

        img = read_float32_tiff(input_path)
        img_down = downsample_half_long_edge(img)
        save_float32_tiff(output_path, img_down)

    print("所有图像已下采样并保存")


if __name__ == "__main__":
    process_directory()
