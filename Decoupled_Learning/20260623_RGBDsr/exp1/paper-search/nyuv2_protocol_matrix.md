# NYUv2 Benchmark 协议矩阵

说明：`明确` 表示论文原文或官方代码直接给出；`统一表` 表示来自 2023 survey
的共同重评测；`待核对` 表示不能仅凭摘要确认，不能直接用于复现实验。

| 方法 | 训练数据 | 测试数据 | LR 生成 | scale | 训练形式 | 指标 / 注意事项 | 证据 |
|---|---|---|---|---|---|---|---|
| GF | 1000/449（统一表） | 449 | survey 统一 bicubic | x4/x8/x16 | classical filter，无神经网络训练 | RMSE；GF 在统一表中是 classical baseline | B |
| DGF | 1000/449（统一表） | 449 | survey 统一 bicubic | x4/x8/x16 | learned guided filter | RMSE；要核对 DGF 原始实现的归一化 | B/C |
| DJF | 1000/449（统一表） | 449 | survey 统一 bicubic | x4/x8/x16 | deep joint filtering | RMSE；survey 统一重评测，不等于原论文数字 | B |
| DJFR | 1000/449（survey 说明沿用） | 449 | survey 统一 bicubic | x4/x8/x16 | deep joint filtering / residual route | RMSE；是历史协议锚点 | B |
| DKN | 官方 NYUv2 split | 官方 test split | 官方代码 / 数据包 | x4/x8/x16 | supervised RGB-guided kernel prediction | RMSE；官方代码的数组命名和测试预处理必须优先 | A |
| FDKN | 官方 NYUv2 split | 官方 test split | 与 DKN 实现相关 | x4/x8/x16 | fast kernel prediction | RMSE；scale-specific 参数 | A/B |
| AHMF | 1000/449（survey unified） | 449 | survey unified protocol | x4/x8/x16 | supervised multimodal CNN | 统一表：1.40 / 2.89 / 5.64 | B |
| DCTNet | 1000 image pairs；官方 README 明确训练图 resize 256x256 | 官方 NYUv2 test set | same preprocessing as DKN/FDSR | x4/x8/x16 | supervised; HDF5 patches/files | 官方 README：x4 1.59155273, x8 3.16303039, x16 5.84125805；不要和 survey unified 行混合 | A |
| JIIF | 论文/代码需逐项核对 | 论文/代码需逐项核对 | 需读官方代码 | x4/x8/x16 | implicit function | survey unified row: 1.37 / 2.76 / 5.27；原始协议待核对 | B/C |
| GraphSR | 论文/代码需逐项核对 | 论文/代码需逐项核对 | 需读官方代码 | x4/x8/x16 | graph regularization | survey unified row: 1.79 / 3.04 / 6.02 | B/C |

## Survey unified reference values

这些数字来自 survey 的共同重评测，适合做文献参考，不代表每篇原论文的原始训练
设置：

| Method | x4 | x8 | x16 |
|---|---:|---:|---:|
| Bicubic | 4.27 | 7.17 | 11.37 |
| GF | 5.05 | 6.97 | 11.13 |
| DGF | 3.21 | 5.92 | 10.45 |
| DJF | 2.80 | 5.33 | 9.46 |
| DJFR | 2.38 | 4.94 | 9.18 |
| DKN | 1.62 | 3.26 | 6.51 |
| FDKN | 1.86 | 3.58 | 6.96 |
| AHMF | 1.40 | 2.89 | 5.64 |
| JIIF | 1.37 | 2.76 | 5.27 |

来源：[Guided Depth Map Super-resolution: A Survey](https://arxiv.org/abs/2302.09598)。

## 与当前工程的关键对齐项

当前工程已经准备了 1000/449，但仍需要把下面各项固定到实验配置中：

1. `split`: sample ID 1-1000 train，1001-1449 test。
2. `scale`: 单独报告 x4、x8、x16，不能混用。
3. `degradation`: 说明 bicubic 的输入/输出尺寸和边界处理。
4. `depth unit`: 记录 PNG raw value、归一化因子和 RMSE 输出单位。
5. `normalization`: 区分 fixed `/10000` 与 per-image min/max。
6. `crop`: 说明是否 crop 6 pixels。
7. `metric`: 统一 RMSE 聚合方式，以及是否使用 valid-depth mask。
