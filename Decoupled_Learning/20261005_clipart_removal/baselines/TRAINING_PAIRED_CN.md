# 三个 JPEG baseline 的配对数据训练

三个新增入口都读取已经生成好的配对数据，不会重新编码 JPEG：

```text
openclipart_5000/jpeg_pairs_512/jpeg_q30/name.jpg  # 输入
openclipart_5000/jpeg_pairs_512/clean_png/name.png # GT
```

训练、验证、测试集合使用上一级目录中的固定 list：

```text
train_list.txt: 3722
val_list.txt:    465
test_list.txt:   466
```

## 训练入口

```text
ARCNN-pytorch/train_paired.py
FBCNN/train_paired.py
SwinIR_downloaded/SwinIR-main/train_paired.py
```

三个脚本都在可见 GPU 数量大于 1 时自动使用 `torch.nn.DataParallel`，输出 `best.pth`、`last.pth` 和 `history.json`。

## 四卡启动

在 Windows PowerShell 中使用对应的 `.ps1` 文件。脚本已经设置：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3
conda run --no-capture-output -n yolo_clean
python -u
```

例如 ARCNN：

```text
baselines/ARCNN-pytorch/run_train_paired.ps1
```

默认都是 Q30。切换到其他质量时，同时修改命令里的：

```text
--quality 10
--quality 20
--quality 50
```

和输出目录，避免覆盖 Q30 checkpoint。

## 模型配置

- ARCNN：RGB，随机 `128x128` patch；protocol160 使用 160 epoch、batch size 8、固定 Adam 学习率。
- FBCNN：RGB，随机 `128x128` patch，固定 QF 输入为 `(100-Q)/100`；protocol160 使用 160 epoch、batch size 8，并在 40/80/120 epoch 衰减。
- SwinIR：RGB，`126x126` patch，窗口大小 7，默认 50 epoch、batch size 4。126 必须能被 7 整除。

SwinIR 官方代码依赖 `timm`。如果 `yolo_clean` 环境没有它，需要先在该环境安装，否则训练入口会在导入模型时失败。

## 注意

这些训练目标是 clean PNG，因此验证的是 JPEG 伪影恢复。它不是自然图像纹理分离实验。所有 baseline 都必须使用同一质量因子、同一 split 和同一评价脚本，才能比较模型结构。
