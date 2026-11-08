# RGB Joint 随机 Patch 实验

配置：`configs/exp3b_joint_patch128.yaml`；入口：`train_joint_refine.py --config <配置路径>`。

## 数据与训练

- 使用原固定 OpenClipart Q30 train/val 列表及已有 JPEG/PNG 配对。
- 每次训练读取一张原图，input/GT 同步均匀随机裁剪 128×128；不重新压缩、不缩放 HR。
- 裁剪后的 input 用原 Lanczos 方法生成 64×64 LR。
- 不做翻转、旋转、颜色增强；与本轮 baseline 的随机裁剪增强方式对齐。
- 验证保持完整 512×512 HR / 256×256 LR，按全图验证 MSE 保存 best。
- 从原 base/refine best 权重重新开始 joint，Adam 状态重新初始化。
- seed 42、batch 8、80 epochs、refine/base LR 0.003/0.0003、StepLR 30/0.5，沿用原 joint。
- 这是 joint 阶段的裁剪与增强对照；前期 base/refine 仍为已有全图训练权重。
- baseline 的损失、优化器及总训练预算仍各不相同。此实验不代表全部训练条件已统一。
- 相比旧 joint，同时改变裁剪和翻转/旋转；因此不能将任何收益单独归因于裁剪。

## 权重路径（相对 clipart 项目根目录）

| 用途 | 路径 |
| --- | --- |
| 原全图 joint best | `train_clipart_joint/ckpt/exp3-b_joint/joint_best.pth` |
| 原全图 joint last | `train_clipart_joint/ckpt/exp3-b_joint/joint_last.pth` |
| 新实验 base 初始化 | `train_clipart_joint/ckpt/exp1-b_conv_base/convguided_exp1b_q30_best.pth` |
| 新实验 refine 初始化 | `train_clipart_joint/ckpt/exp2-b_refine_cached_aug/refine_exp2b_best.pth` |
| 新 patch joint best（训练后生成） | `train_clipart_joint/ckpt/exp3-b_joint_patch128/joint_best.pth` |
| 新 patch joint last（训练后生成） | `train_clipart_joint/ckpt/exp3-b_joint_patch128/joint_last.pth` |

新目录还会保存 `history.jsonl` 与 `config_used.yaml`。
只在同样的裁剪/增强设置下使用新目录中的 last 续训。

## 全图推理

参数结构不变，新权重可用于完整 512×512 全图推理。
现有 `evaluate_all_models.py` 中 `exp3-b` 仍指向原全图权重；评测新实验时需显式选择新权重，并使用独立预测输出目录。
