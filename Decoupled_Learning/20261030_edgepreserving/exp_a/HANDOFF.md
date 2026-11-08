# 实验 A 交接说明

## 实验范围

这是针对 20260818 WLS 保边数据的四个单通道、从头开始、端到端联合训练模型。guided filter 数学算子和三阶段递推保持不变，变化集中在 LR predictor、stage 内部 block 和 loss 组织方式。

训练器采用单机 NCCL DDP。每个 rank 先绑定 `LOCAL_RANK` 对应 GPU，再初始化
process group；DataLoader 使用 `pin_memory=False`。默认四卡、每卡 batch 4、全局
batch 16、BF16、HR patch `128x128`、学习率 `3e-4`。
启动脚本默认保持 `NCCL_SHM_DISABLE=0`，与 handoff 的标准 NCCL 启动方式一致；
只有在确认共享内存 transport 本身失败时，才通过环境变量显式设置为 `1` 重试。
本机 `/dev/shm` 当前为 32 GB，前次关闭 SHM 后错误转为 CUDA host allocation
失败，因此不能把 `NCCL_SHM_DISABLE=1` 当成默认修复。

## 当前状态

四个实验的代码已经实现，但没有声称新的训练指标已经完成。C0/C1 保留历史
delta-space loss 作为对照；如果当前判断是该 loss 不适合作为新方案，主实验应先跑
A0/A1，不能把 C0/C1 当作推荐训练目标。

之前训练好的 exp7C/C0 checkpoint 已复制到 `checkpoints/c0/`，来源记录在 `checkpoints/c0/PROVENANCE.json`。新训练器不会加载这些权重。

当前已验证：

- 训练和评估脚本语法检查通过；
- 四个 variant 的 forward 输出尺寸一致；
- 真实数据上的 LR/HR crop 检查通过；
- A1 的 CUDA forward/backward 检查通过；
- 评估脚本使用全图，只写 CSV/JSON 指标。

## 四个实验

- `c0`：原始 LR predictor、原始 stage 结构、与原 edge-preserving/exp7C 对齐的 delta-space loss，加上 `0.5 * SmoothL1(coarse0, GT)` coarse fidelity 项；edge mask 使用逐图 Otsu。
- `c1`：VIV-inspired 3x3/9x9 LR predictor，其他设置与 C0 相同。
- `a0`：原始 LR predictor、原始 stage 结构，使用统一 image-space loss，监督 `coarse0/coarse1/coarse2/final`，权重为 `0.05/0.10/0.30/0.60`。
- `a1`：VIV-inspired LR predictor、NAF-style stage blocks，以及 A0 的统一 loss。

## 训练命令

```bash
cd /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261030_edgepreserving
CUDA_VISIBLE_DEVICES=0,1,2,3 VARIANT=a1 MAX_STEPS=20 bash exp_a/run_4gpu_ddp.sh
```

确认四 rank smoke test 成功后，正式跑 A0 和 A1 时分别设置 `VARIANT=a0`、
`VARIANT=a1` 并移除 `MAX_STEPS`。不要把 C0/C1 的 delta-space loss 当成推荐目标。

四卡 NCCL DDP（建议先用 128 HR patch 做短跑）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash exp_a/run_4gpu_ddp.sh
```

可通过环境变量覆盖：`VARIANT`、`CROP_SIZE`、`BATCH_SIZE_PER_GPU`、`EPOCHS`、
`LR`、`PRECISION`、`NUM_WORKERS`、`MASTER_PORT`、`MAX_STEPS`。例如设置
`MAX_STEPS=20` 可用相同 NCCL DDP 路径执行短 smoke test；正式训练保持默认 `0`。

如果 WSL NCCL 持续因 `/dev/shm/nccl-*` 创建失败，可以直接用 WinTrain 命令调用
`run_sequence_windows_ddp_gloo.py`。它依次训练 A1、C1、A0，使用四进程 DDP/Gloo，
不使用 DataParallel。先用 `--max-steps 20` 短跑，再去掉该限制正式训练。Windows
默认 workers 为 0。Gloo 和 DataParallel 的速度需要实测比较。

## 评估命令

```bash
python -u exp_a/src/evaluate_exp_a.py \
  --checkpoint exp_a/runs/a1/final.pth \
  --data-root /mnt/d/ResearchCode/S2R-HDR/data/20260818 \
  --output-dir exp_a/evaluation/a1 --device cuda:0
```

训练使用 paired LR/HR 随机 crop；测试使用完整图像和偶数边裁剪，不能把训练 crop 协议当成推理协议。
128 patch 只改变训练阶段的上下文和吞吐，不改变全图推理接口；是否损失全局结构
必须以四个 variant 的全图测试指标验证。
