import os
import json
import numpy as np
from PIL import Image
import glob
import cv2

def read_bin_image(file_path, width, height):
    # 读取 bin 文件并转换为 16 位图像
    with open(file_path, 'rb') as bin_file:
        data = np.fromfile(bin_file, dtype=np.uint16)
    image = data.reshape((height, width, 3))
    return image

def apply_ccm(image, ccm):
    # 应用颜色校准矩阵
    ccm = np.array(ccm).reshape((3, 3))
    image = np.dot(image, ccm.T)
    return image

def save_tiff_image(image, output_path):
    # 将图像数据转换为32位定点数格式（范围0-1）
    #image = (image * (2**32 - 1)).astype(np.uint32)
    # 使用OpenCV保存图像
    cv2.imwrite(output_path, image[:,:,::-1], ((int(cv2.IMWRITE_TIFF_COMPRESSION), 1)))

def find_json_file(root, base_name):
    # 使用glob找到符合条件的json文件
    pattern = os.path.join(root, f"{base_name}*.json")
    json_files = glob.glob(pattern)
    return json_files[0] if json_files else None

def process_files(root_path, out_path):
    for root, _, files in os.walk(root_path):
        for file in files:
            if '_16bitLogRGB' in file and file.endswith('.bin'):
                base_name = file.split('_')[0]
                json_file = find_json_file(root, base_name)

                if json_file:
                    with open(json_file, 'r') as f:
                        json_data = json.load(f)
                        ccm = json_data.get('BpsParam', {}).get('CCM', None)
                        if ccm is None:
                            print(f"CCM not found in {json_file}")
                            continue

                    # 获取图像的宽高
                    resolution = file.split('_')[-1].split('.')[0]
                    width, height = map(int, resolution.split('x'))

                    # 读取并处理 bin 文件
                    bin_file_path = os.path.join(root, file)
                    image = read_bin_image(bin_file_path, width, height)

                    # 对图像进行线性转换 // 线性化
                    image_linear = np.exp(image / 1024.0)
                    
                    # Normalize
                    #image_linear = image_linear/np.max(image_linear)
                
                    # 应用颜色校准矩阵
                    processed_image = apply_ccm(image_linear, ccm)
                    processed_image = np.maximum(processed_image, 0.0)
                    #print(processed_image)
                    # 保存处理后的图像
                    output_file = os.path.join(out_path, f"{base_name}_NormalizedHDR.tiff")
                    save_tiff_image(processed_image.astype(np.float32), output_file)
                    print(f"Processed and saved image: {output_file}")
                else:
                    print(f"JSON file not found: {json_file}")

# 设置路径并执行处理
in_path = '/mnt/d/nuoqi/S2R-HDR/data/origin_data'
out_path = '/mnt/d/nuoqi/S2R-HDR/data/test_vivo/hdrimg'

process_files(in_path, out_path)

