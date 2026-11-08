# Exp3 Joint Fine-tuning (Exp7C)

## 初始化与数据流

- Exp3-a 从 `exp1-a_conv_base` best 和 `exp2-a_refine_cached_aug` best 初始化。
- Exp3-b 从 `exp1-b_conv_base` best 和 `exp2-b_refine_cached_aug` best 初始化。
- Joint 阶段在线从 JPEG 构造 LR/HR，并运行可训练 base 得到 coarse；不读取 `openclipart_5000/coarse_cache`。
- 训练增强在 base 前同步应用于 LR、HR、clean GT，使用水平/垂直翻转和 90 度倍数旋转。
- 训练和验证使用已有固定 train/val list；test list 不参与训练。

## Exp7C 设置

- Refine Adam learning rate: `3e-3`。
- ConvGuided base Adam learning rate: `3e-4`。
- StepLR: 每 30 epochs 乘 `0.5`，80 epochs。
- Cascade loss: `0.1*l1 + 0.3*l2 + 0.6*l3`，Otsu edge mask，`dilate=0`，L1。
- Coarse fidelity loss: `0.5 * SmoothL1(coarse, clean)`。
- 两组均读取 YAML 配置，batch 8、workers 16。

## 输出

- `ckpt/exp3-a_joint/joint_best.pth` 和 `joint_last.pth`
- `ckpt/exp3-b_joint/joint_best.pth` 和 `joint_last.pth`
- 同目录的 `history.jsonl` 与 `config_used.yaml` 保存逐 epoch 记录和实际配置。
- checkpoint 同时保存 base、Refine、optimizer、scheduler，可用 `--resume` 接续。

## 运行

在 Windows `wintrain` 环境执行 `train_joint_refine.py --config <配置 YAML>`。
PyYAML 需在训练环境可导入。Joint 训练会重新计算 coarse，不会把 coarse cache 当成固定输入。
