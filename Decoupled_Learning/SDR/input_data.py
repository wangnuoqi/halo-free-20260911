import os
import random
import shutil

def copy_random_images_with_suffix(source_dir, target_dir, num_images=100, suffix="-input"):
    """
    从源目录随机抽取图片并添加后缀复制到目标目录
    
    参数:
        source_dir: 源图片目录路径
        target_dir: 目标目录路径
        num_images: 要抽取的图片数量(默认100)
        suffix: 要添加的后缀(默认"-input")
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取源目录下所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    all_images = [f for f in os.listdir(source_dir) 
                if f.lower().endswith(image_extensions)]
    
    # 检查是否有足够图片
    if len(all_images) < num_images:
        raise ValueError(f"源目录只有 {len(all_images)} 张图片，少于要求的 {num_images} 张")
    
    # 随机选择图片
    selected_images = random.sample(all_images, num_images)
    
    # 复制并重命名文件
    for img in selected_images:
        # 分离文件名和扩展名
        name, ext = os.path.splitext(img)
        # 构建新文件名
        new_name = f"{name}{suffix}{ext}"
        # 完整路径
        src_path = os.path.join(source_dir, img)
        dst_path = os.path.join(target_dir, new_name)
        
        # 复制文件
        shutil.copy2(src_path, dst_path)
    
    print(f"成功复制 {num_images} 张图片到 {target_dir}")

# 使用示例
if __name__ == "__main__":
    source_directory = "./VOCSegmentation/VOCdevkit/VOC2012/JPEGImages"
    target_directory = "./VOC2012_input"
    
    copy_random_images_with_suffix(source_directory, target_directory)