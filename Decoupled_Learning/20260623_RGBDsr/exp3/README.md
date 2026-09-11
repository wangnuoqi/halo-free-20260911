# Exp3：origin 与 RF 联合训练

当前 B/C 的正式新入口是 `train_exp3_BC_freeze_joint.sh`：完整训练池训练，每个 epoch 都在 449 张测试集上评估并按测试 RMSE 保存最优权重（这是本项目当前的分数优化实验口径，不是无泄漏 benchmark）。每个方法分别运行 frozen（只训练 RF）和 joint（origin+RF 解冻）两组。

joint 分支沿用 Exp7C 的 origin fidelity 约束，并加入各 origin 的原始任务 loss：DKN 为 `Exp6D cascade + 0.5 * SmoothL1(origin_coarse,target) + 1.0 * L1(origin_coarse,target)`，DCTNet 将最后一项换为 `MSE(origin_coarse,target)`。这些 origin loss 只在解冻时启用；frozen 分支保持原有 RF loss。

- `A_convguided_joint`：本项目全图 NYUv2、完整 1000 张训练池、80 epoch 的 ConvGuided + RF 联合训练协议。
- `B_dkn_joint_official`：使用 DKN 官方 1000/449 数组、batch=1、PIL bicubic 和 DKN 的 x8 origin 输入。
- `C_dctnet_joint_official`：使用 DCTNet 官方风格的 256×256 patch、batch=8、RGB mean/std 归一化和 x8 origin 输入。

主实验不再单独重训 DKN/DCTNet origin；因为官方 checkpoint 已完成输入协议和结果核验，直接作为联合训练初始化。

B/C 的 origin checkpoint 只作为初始化；frozen 模式保持 origin 不变，joint 模式与 RF 一起更新。四组输出彼此隔离。

DKN 使用完整 `train_depth_split.npy` 训练、`test_depth.npy` 测试。DCTNet 必须提供官方 processed H5/测试目录（脚本中的 `WIN_DCT_TRAIN`、`WIN_DCT_TEST`），否则不会启动。

ConvGuided joint 使用 `L1(origin_coarse,target)` origin loss，origin lr 为 `3e-4`、RF lr 为 `3e-3`；旧 A checkpoint（900 张训练、未加入 origin loss）不复用。

最终主实验为三组：A=ConvGuided 联合训练，B=官方 DKN origin+RF 联合训练，C=官方 DCTNet origin+RF 联合训练。唯一运行入口为 `train_exp3_ABC_x8.sh`；`train_origin_official.py` 仅保留为可选 origin 审计，不属于主实验链路。
