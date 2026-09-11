# Exp9：coarse loss 消融实验

Exp9 保留 DnCNN/FFDNet 官方 patch 数据协议和 Exp7C RF，只比较联合微调时
约束 coarse output 接近 clean target 的损失。ConvGuided 是自定义 coarse
模型，因此不纳入这个消融实验。

| 实验 | coarse 项 | 联合总损失 |
| --- | --- | --- |
| `*_mse_smooth_patch` | `MSE(coarse, clean) + 0.5 * SmoothL1(coarse, clean)` | `RF + MSE + 0.5 * SmoothL1` |
| `*_mse_patch` | `MSE(coarse, clean)` | `RF + MSE` |

DnCNN 和 FFDNet 原始 denoiser loss 是 masked MSE。Exp9 不修改 Exp7/Exp8
默认逻辑；现有联合训练默认仍为 `RF + 0.5 * SmoothL1`。

每个网络使用一条命令，单卡串行运行两个变体：

```bash
GPU_ID=0 bash /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260621_denoiseretrain/exp9/run_exp9_dncnn_single_gpu_wintrain.sh
GPU_ID=1 bash /mnt/d/nuoqi/halo-free/Decoupled_Learning/20260621_denoiseretrain/exp9/run_exp9_ffdnet_single_gpu_wintrain.sh
```

结果保存在 `exp9/formal_v1/runs/`。已完成的实验会跳过，未完成的输出目录
会在重新启动前删除。
