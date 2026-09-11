import os
import re
from pathlib import Path

def extract_input_prefix(filename):
    """从output文件名中提取2007_000027-input格式的前缀"""
    match = re.match(r'^(\d+_\d+-input)', filename)
    return match.group(1) if match else None

def compare_and_clean():
    # 设置路径
    base_path = "juicefs_sharing_data/11181409/DecoupleLearning"
    input_dir = os.path.join(base_path, "VOC2012_L0smooth_input")
    output_dir = os.path.join(base_path, "VOC2012_L0smooth_output/VOC2012_L0smooth_output")
    # 步骤1：收集output文件夹的所有有效前缀
    output_prefixes = set()
    for filename in os.listdir(output_dir):
        if filename.endswith('.jpg'):
            prefix = extract_input_prefix(filename)
            if prefix:
                output_prefixes.add(prefix)
    
    print(f"找到 {len(output_prefixes)} 个output文件前缀")
    
    # 步骤2：遍历input文件夹并删除不匹配的文件
    deleted_count = 0
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):
            # 检查文件名是否以output中的某个前缀开头
            matched = any(filename.startswith(prefix) for prefix in output_prefixes)
            
            if not matched:
                file_path = os.path.join(input_dir, filename)
                print(f"准备删除: {file_path}")
                # 实际删除（取消注释以下两行）
                #os.remove(file_path)
                deleted_count += 1
    
    print(f"操作完成！共标记 {deleted_count} 个需要删除的文件")

if __name__ == "__main__":
    compare_and_clean()