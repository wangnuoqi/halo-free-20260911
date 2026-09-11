# 对齐建议

## 推荐的第一批实验

为了回答“guided filter baseline 的论文通常怎么训练”，建议先固定一条最小可比
protocol：

```text
train: sample ID 1-1000
test:  sample ID 1001-1449
HR:    640x480
LR:    bicubic degradation for x4/x8/x16
input: LR depth + HR RGB
loss:  L1（survey 的 unified deep-model description）
optimizer: Adam
metric: RMSE
report: no-crop 和 crop-6 分开报告
```

先跑以下顺序：

1. Bicubic：验证 degradation 和 RMSE 代码。
2. GF：无训练，作为 classical guided-filter baseline。
3. DGF：验证 deep guided filter 路线。
4. DJF / DJFR：历史 guided-filter/deep joint filtering 基线。
5. DKN / FDKN：公开 kernel-prediction 基线。
6. 你们自己的 RGB-guided 模型。

## 训练和评测不能混在一起

- GF 没有神经网络训练过程；它只需要在每个测试样本上运行滤波器。
- DGF、DJF、DJFR、DKN、FDKN 等需要用 1000 个训练样本学习参数。
- 论文中“1000 image pairs”不等于“1000 个训练 patch”。很多代码会从 1000
  张图裁剪、缩放或生成大量 patch/HDF5 样本。
- 测试仍然应该使用 449 个完整 image pairs，而不是从测试图随机裁 patch。

## 当前工程应重点核对

1. 你们 `benchmark_train_1000` 中的 patch 数量不能直接写成训练图片数量。
2. `dkn_nyuv2_1000_449` 是 DKN 数组格式，不是另一份独立 benchmark。
3. 你们的固定 `/10000` normalization 和 legacy per-image min/max 是两种不同协议，
   结果必须分开命名。
4. survey 的 RMSE 是统一重评测值；DCTNet README 的数字是官方实现输出，不能
   放在同一列而不标注来源。
5. 最终论文表格至少添加 `protocol_match` 列，标记 exact / partial / unknown。

## 下一步逐篇核对顺序

建议按下面顺序下载和读原文/代码：

1. DKN 官方仓库和论文。
2. DJFR 原文和官方实现。
3. DGF 原文和实现。
4. DCTNet README、训练代码和测试预处理。
5. AHMF 及 survey training code。
6. JIIF、FDKN。

每篇最终要填的字段是：

```text
paper / method
venue / year
NYUv2 split
number of training images
patch extraction
HR/LR resolution
LR degradation
depth normalization / unit
loss / optimizer / schedule
augmentation
test crop / mask
RMSE aggregation
official checkpoint or re-training
```
