import numpy as np
import os
import cv2

folder_path = '/data/vjuicefs_ai_camera_llm/intern_data/nuoqi/DecoupleLearning/VOC/VOC_edge_V7_test_label_r2'

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith('.png'):
            full_path = os.path.join(root, file)
            pic = cv2.imread(full_path)
            edge_map = pic.astype(np.float32) / 255.0

print('已经看到了狗熊')