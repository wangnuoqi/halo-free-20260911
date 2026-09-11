# Survey NYUv2 数据流记录

这里的 `min-max` 始终指 depth，不指 RGB。

## Survey 训练流

训练池是前 1000 对 RGB-D。官方 dataloader 默认按 `val_ratio=0.1` 再切成
900 对 train 和 100 对 validation。

```text
HR RGB + HR depth label
  -> 同一坐标同步 random crop
  -> RGB patch 256x256 + HR depth patch 256x256
  -> 对 HR depth patch 做 bicubic downsample
     x4: 256x256 -> 64x64
     x8: 256x256 -> 32x32
     x16: 256x256 -> 16x16
  -> 输入：LR depth + HR RGB
  -> RGB /255；训练 depth 不做 min-max
  -> label：HR depth patch；loss：L1
```

几何增强同步作用于 RGB、LR depth 和 HR depth：horizontal flip、vertical
flip、transpose/90-degree rotation。

## Survey validation / final test 流

两者都使用完整图像，不做 random crop；但官方代码中的 validation 和 final
test 的指标处理并不完全相同。

```text
完整 HR depth
  -> 只对当前 depth 图计算 min/max
  -> depth_norm = (depth - min) / (max - min)
  -> bicubic downsample 得到 LR depth
HR RGB -> RGB / 255
模型输出 [0,1]
  -> pred = pred * (max - min) + min
  -> final test 时 prediction 和 GT 同时 crop 6 pixels
  -> RMSE
```

`min/max` 只来自当前这张 GT depth 图；RGB 不参与 min-max。

| 阶段 | RGB | depth input | depth label / metric |
|---|---|---|---|
| train | `/255` | HR depth 生成 LR，不做 min-max | 原始预处理尺度 HR depth |
| validation | `/255` | 完整 GT depth per-image min-max 后 bicubic | 官方 `val()` 使用 `border=0, is_train=True`，RMSE 函数跳过 inverse/crop |
| final test | `/255` | 完整 GT depth per-image min-max 后 bicubic | inverse min-max 后 crop-6 RMSE |

## 依据

论文 PDF：

- 第 24 页，Section 7 `Experiments` -> `Datasets and Metrics`：前 1000 对训练，
  后 449 对 NYUv2 测试。
- 第 25 页：NYUv2 使用 Bicubic 生成 LR depth，scale 为 x4、x8、x16，并使用
  RMSE。
- 第 26 页，Table 7：统一 NYUv2 结果表。

本地源码：

- `docs/papers/Guided Depth Map Super-resolution A Survey.tex:956-958`
- `docs/papers/Guided Depth Map Super-resolution A Survey.tex:1004-1032`

官方代码：

- `code/data/nyu.py`：同步 crop、训练增强、从 HR depth patch 生成 LR；测试完整
  图像 per-image depth min-max 后生成 LR。
- `code/utility.py`：RGB `/255`；`root_mean_sqrt_error` 只有 `is_train=False`
  时才用 GT depth min/max inverse；RMSE 支持 border crop。
- `code/trainer.py`：`val()` 调用 `border=0, is_train=True`；final `test()` 调用
  `border=6, is_train=False`。
- `code/option.py`：默认 `val_ratio=0.1`、`patch_size=256`、`batch_size=32`、
  `loss=1*L1`、`optimizer=Adam`。

官方链接：[nyu.py](https://github.com/zhwzhong/Guided-Depth-Map-Super-resolution-A-Survey/blob/main/code/data/nyu.py)、[utility.py](https://github.com/zhwzhong/Guided-Depth-Map-Super-resolution-A-Survey/blob/main/code/utility.py)、[trainer.py](https://github.com/zhwzhong/Guided-Depth-Map-Super-resolution-A-Survey/blob/main/code/trainer.py)、[option.py](https://github.com/zhwzhong/Guided-Depth-Map-Super-resolution-A-Survey/blob/main/code/option.py)。

## 当前工程的实际状态

当前 [train_convguided_base.py](../train_convguided_base.py) 使用
[nyuv2_full_dataset.py](../nyuv2_full_dataset.py) 读取完整图：900 张训练、100 张
validation，以及独立的 449 张 final test。训练不再读取旧的 patch 目录。

Dataset 对每张 depth 做 full-image min-max，训练 `.npy` 的米单位在保存 stats 前
转换为毫米，RGB 统一 `/255`，再从 normalized HR depth 用 PIL bicubic 生成 x4 LR。
训练只做同步 flip/transpose；validation/test 不做增强。

模型输出保持在 `[0,1]`，validation/test 用对应图像的 min/max inverse；final test
另外 crop 6 后计算 RMSE。训练 loss 保持当前 L1、Sobel 和 consistency 组合。

旧的 [eval_legacy_nyuv2.py](../eval_legacy_nyuv2.py) 仍可作为独立 final-test
复核脚本，但训练脚本已经不再把训练 patch 当作 test，也不再使用固定 `/10000`
normalization。
