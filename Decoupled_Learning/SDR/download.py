from torchvision.datasets import VOCSegmentation
import torchvision.transforms as transforms

# 设置数据集路径和参数
root = './VOCSegmentation'  # 数据集将下载到此目录
year = '2012'    # 年份（支持 '2007' 或 '2012'）
image_set = 'train'  # 可选 'train', 'val', 'trainval'

# 下载数据集
dataset = VOCSegmentation(
    root=root,
    year=year,
    image_set=image_set,
    download=True,  # 关键：启用自动下载
    transform=transforms.ToTensor()  # 可选：图像转换
)