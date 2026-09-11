import os
import cv2
import tifffile

folder_path = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_SDRtest_label'
output_file = 'vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/HDR_wls_SDRretest512_list.txt'


with open(output_file, 'w') as f:
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            #根据路径下需要读取的文件改写
            if file.lower().endswith('.png'):
                # 读取图片尺寸
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)
                #img = tifffile.imread(file_path)
                # 处理文件名：替换-为_，并添加尺寸
                base_name = os.path.splitext(file)[0]
                #new_name = f"{base_name.replace('-', '_')}_{w}_{h}.jpg"
                # 写入完整路径（保持原始路径结构，只修改文件名）
                #rel_path = os.path.relpath(root, folder_path)
                output_path = os.path.join(folder_path, base_name+".png")
                #output_path = output_path.replace('\\', '/')  # 统一使用正斜杠
                
                f.write(f"{output_path}\n")  # 原始路径和转换后路径用|分隔
            else:
                print(f"警告：无法读取图片尺寸 {file_path}")

print(f"文件路径和转换后的名称已成功写入 {output_file}")