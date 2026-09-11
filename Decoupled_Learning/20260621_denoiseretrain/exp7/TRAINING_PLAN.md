# Exp7 Post-Base Training Plan

## Goal

在统一的全图 sigma25 协议下，分别比较三个本地 base denoiser 后接 RF 时的
冻结训练与联合训练。所有正式产物位于 `exp7/formal_v2/`。

## Phase 0: Train And Accept Bases

先运行 `run_exp7_bases_four_gpu_wintrain.sh`。该命令依次以四卡
DataParallel 训练下列 base：

| Model | Initial State | Output |
|---|---|---|
| ConvGuided | random initialization | `models/convguided_base_full/convguided_best.pth` |
| DnCNN | `models/init/dncnn_sigma25_source.pth` | `models/dncnn_base_full/dncnn_best.pth` |
| FFDNet | `models/init/ffdnet_sigma25_source.pth` | `models/ffdnet_base_full/ffdnet_best.pth` |

所有 base 使用：BSDS400 完整图、8-way 几何增强、随机 NumPy sigma25 AWGN、
batch16、160 epoch、Adam lr `1e-4`、第60/120 epoch衰减0.5。每个 epoch 在
完整 BSD68、固定 `RandomState(0)` AWGN 上验证，并保存最佳 PSNR checkpoint。

验收条件：三个输出目录都包含 `manifest.json` 与对应 `*_best.pth`；日志中应能
看到 `torch.cuda.device_count=4`。若任一 base 失败，不启动其 RF 分支。

## Phase 1: Frozen RF Baseline

在每个已验收 base 后训练一组 frozen 分支：

```text
clean full image -> AWGN -> frozen local base -> coarse -> Exp6D RF -> final
```

三组输出：

```text
runs/exp7B_convguided_frozen_exp6d_full/
runs/exp7B_dncnn_frozen_exp6d_full/
runs/exp7B_ffdnet_frozen_exp6d_full/
```

超参统一为：全图、8-way、batch16、160 epoch、RF Adam lr `3e-3`、第60/120
epoch衰减0.5。损失仅为 Exp6D 三阶段 RF edge loss；base 不接收梯度。

## Phase 2: Joint RF Training

对同一份本地 base 分别训练 joint 分支：

```text
clean full image -> AWGN -> trainable local base -> coarse -> Exp7C RF -> final
```

三组输出：

```text
runs/exp7A_convguided_joint_exp7c_full/
runs/exp7A_dncnn_joint_exp7c_full/
runs/exp7A_ffdnet_joint_exp7c_full/
```

除模型状态与 RF 类型外，数据、batch、epoch、scheduler 与 Phase 1 完全相同。
优化器有两个参数组：RF lr `3e-3`，denoiser lr `1e-4`。每个 batch 的总损失为：

```text
L = L_Exp7C_RF + 0.5 * SmoothL1(coarse, clean)
```

该 coarse fidelity 项必须始终记录，不能只训练 RF final 输出。

## Phase 3: Evaluation And Reporting

每个训练目录自动在每 epoch 测试 BSD68，并按 final PSNR 保存最佳 bundle。训练
结束后，对六个 best bundle 使用 `../evaluate_bundle.py` 重新评测并写入：

```text
results/exp7A_convguided.json
results/exp7A_dncnn.json
results/exp7A_ffdnet.json
results/exp7B_convguided.json
results/exp7B_dncnn.json
results/exp7B_ffdnet.json
```

每条结果必须包含 base/coarse PSNR、final PSNR、delta、source epoch、RF 类型、
冻结状态和噪声协议。正式比较只使用这些 JSON，不使用 legacy `exp7/runs/` 结果。

## Execution Order

1. 四卡顺序完成三个 base。
2. 检查三个 best base 和 BSD68 coarse PSNR。
3. 依次以四卡运行 DnCNN frozen、DnCNN joint。
4. 依次以四卡运行 FFDNet frozen、FFDNet joint。
5. 依次以四卡运行 ConvGuided frozen、ConvGuided joint。
6. 重新评测六个 best bundle，汇总同一 BSD68 协议下的 coarse/final 对比。

这样每一个训练脚本均独占 GPU0--3；不会发生多实验抢卡，也不会把不同 base
或不同 Exp 的 checkpoint 混用。
