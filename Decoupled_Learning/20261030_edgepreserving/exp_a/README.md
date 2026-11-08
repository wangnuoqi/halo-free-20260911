# 实验 A：VIV-LR 与统一联合训练

这是 20261030 保边滤波实验的独立实现，包含四个单通道、从头开始联合训练的模型。训练支持 WSL 下的单机四卡 NCCL DDP。

| 编号 | LR predictor | Stage block | Loss / 训练方式 | 定位 |
|---|---|---|---|---|
| `c0` | 原始膨胀卷积 LR 网络 | 原始 stage 结构 | 原始 delta-space joint loss | 保守基线 |
| `c1` | VIV-inspired LR 网络 | 原始 stage 结构 | 原始 delta-space joint loss | 保守替换方案 |
| `a0` | 原始膨胀卷积 LR 网络 | 原始 stage 结构 | 统一 image-space cascade loss | 新 loss 基线 |
| `a1` | VIV-inspired LR 网络 | NAF-style stage blocks | 统一 image-space cascade loss | 完整改进方案 |

若沿用“C 的 delta-space loss 不适合作为新方案”的判断，`c0/c1` 只作为历史目标
对照，不应作为主训练方案；主比较用 `a0` 和 `a1`。

四个实验都保留 ConvGuided 的局部 box statistics、局部 affine reconstruction 和三阶段 `delta/a/b/gate` 递推。训练器不会加载旧 checkpoint。历史 C0/exp7C 权重仅作为参考，复制在 `checkpoints/c0/` 下。

## C 与 A 的 loss 区别

两组实验的 loss 形式不同，不只是权重不同。

### C 系列：历史 delta-space joint loss

```text
target_delta = GT - coarse0
L_delta = 0.10 * edge_loss(delta1, target_delta)
         + 0.30 * edge_loss(delta2, target_delta)
         + 0.60 * edge_loss(delta3, target_delta)
L_C = L_delta + 0.5 * SmoothL1(coarse0, GT)
```

edge mask 从 `target_delta` 计算，使用原始代码的逐图 Otsu 梯度阈值、无边缘 fallback、平滑区/边缘区像素误差和边缘 Sobel 梯度误差。

### A 系列：统一 image-space cascade loss

```text
L_A = 0.05 * edge_loss(coarse0, GT)
    + 0.10 * edge_loss(coarse1, GT)
    + 0.30 * edge_loss(coarse2, GT)
    + 0.60 * edge_loss(final, GT)
```

每项都在图像空间计算，edge mask 从 GT 计算，并使用同一套逐图 Otsu 阈值、无边缘 fallback、平滑区域像素误差、边缘像素误差和边缘梯度误差。

## 代码位置

- 新训练器：`src/train_exp_a.py`
- 全图指标评估：`src/evaluate_exp_a.py`
- 四个配置：`configs/c0.json`、`configs/c1.json`、`configs/a0.json`、`configs/a1.json`
- 原始代码快照：`source_snapshot/`
- 历史 C0 权重：`checkpoints/c0/`
- 四卡 NCCL 启动脚本：`run_4gpu_ddp.sh`
- Windows 原生四卡 Gloo DDP：`run_sequence_windows_ddp_gloo.py`
- Windows 原生四卡 DataParallel：直接调用 `src/train_exp_a.py --parallelism dp`

## 训练几何与速度

训练使用随机 paired crop，默认 HR patch 为 `128x128`，对应 LR patch 为
`64x64`；测试仍然使用完整图像。相对于 `256x256`，128 patch 的单步激活量约
为四分之一，通常能明显提高吞吐或允许增大每卡 batch，但它看到的上下文更小，
最终指标必须用同一套全图评估确认。

四卡默认每卡 batch 为 4，全局有效 batch 为 16。这里不能直接照搬 VIVNet 的
`lr=1e-3`，因为本实验的模型和 edge-aware loss 不同；当前默认 `3e-4` 是新
from-scratch joint 模型的保守起点。正式训练默认 200 epochs，学习率从 `3e-4`
按 cosine decay 降到 `3e-6`；DP 和 DDP 使用同一调度。若保持全局 batch 不变而只
改变 micro-batch，学习率不需要按卡数线性放大。

这与旧 exp7C 不是同一训练协议：旧脚本使用全图训练、batch size 等于当时可见
GPU 数、80 epochs、Adam 分组学习率（base `3e-4`、refine `3e-3`）和每 30 epochs
将学习率乘 `0.5` 的 StepLR。新脚本采用 paired patch、全局 batch 16、AdamW
`3e-4`、200 epochs cosine decay；新结果不是 exp7C 的严格复现。

训练器每个 epoch 保存 `latest.pth`，结束保存 `final.pth`；目前没有验证集模型
选择，因此不会产生 best checkpoint。训练器也没有 optimizer-state resume。

## C1：从 exp7C 初始化，只替换 LR predictor

`--variant c1 --init-exp7c` 是迁移微调实验，不是 from-scratch C1：它从
`checkpoints/c0/` 严格加载 exp7C 的 ConvGuided 和 3-stage（51/51 个 state tensors），
保留新建的 VIV-inspired LR predictor。LR predictor 最后使用 sigmoid，将输出限制到
旧 LR predictor 使用的 `[0,1]` 范围；其输出层以接近 0.5 的小权重初始化，避免初始
输出全落在 clamp 的零梯度区。LR predictor 使用 `3e-4`，载入的 ConvGuided 和 stage
使用 `3e-5` 微调学习率；两个组都按相同 cosine schedule 衰减。原 guided-filter clamp
和 refine 坐标流程不变。

四卡 NCCL 正式训练命令（HR patch 128，每卡 batch 16，全局 batch 64）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 VARIANT=c1 INIT_EXP7C=1 \
OUTPUT_DIR=/mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261030_edgepreserving/exp_a/runs/c1_vivlr_exp7c_transfer_ddp4_patch128_b64_fixinput \
BATCH_SIZE_PER_GPU=16 CROP_SIZE=128 EPOCHS=200 LR=3e-4 FINETUNE_LR=3e-5 \
MIN_LR=3e-6 LR_SCHEDULE=cosine PRECISION=bf16 NUM_WORKERS=0 \
bash exp_a/run_4gpu_ddp.sh
```

该脚本使用四进程 `DistributedDataParallel`、NCCL、`DistributedSampler`，并保持
`pin_memory=False`。新 run 会另存于指定目录，不覆盖此前的 from-scratch C1。训练后
使用该目录的 `final.pth` 重新运行全图测试。

注意：2026-10-30 首次迁移训练目录 `c1_vivlr_exp7c_transfer_ddp4_patch128_b64`
是在 `Full.forward()` 对已中心化 HR 输入再次减 `0.5`、stage Sobel 特征未按旧代码
归一化时训练的，结果无效。修正后的实现已经通过旧 exp7C 逐图复现测试，PSNR
`47.114087647 dB`，与旧记录一致；请使用上面的 `_fixinput` 新目录重新训练。

## Windows 原生 DDP

Windows 原生运行时使用 Gloo 进程组；脚本仍启动四个
`DistributedDataParallel` 进程和 `DistributedSampler`，不是 `DataParallel`。
它绕开 WSL 的 NCCL `/dev/shm` 问题，但通信速度必须用实测数据判断，不能预先保证
一定快于 DataParallel 或 Linux NCCL。Windows 默认 `NumWorkers=0`，先保证启动稳定。
Gloo 对 CUDA tensor 的 collective 会经 CPU pinned buffer 中转；它可能减轻 DP 的主卡
scatter/gather 瓶颈，也可能被 CPU 中转与通信开销抵消，因此速度要用相同步数实测。

直接用 WinTrain 启动 A1、C1、A0 的 Windows 原生 Gloo DDP 串行任务。先做 20 步 smoke test：

```bash
wintrain '$env:CUDA_VISIBLE_DEVICES="0,1,2,3"; $env:PYTHONUNBUFFERED="1"; conda run --no-capture-output -n yolo_clean python -u D:\ResearchCode\halo-free\Decoupled_Learning\20261030_edgepreserving\exp_a\run_sequence_windows_ddp_gloo.py --variants a1 c1 a0 --data-root D:\ResearchCode\S2R-HDR\data\20260818 --batch-size-per-gpu 8 --crop-size 128 --epochs 200 --lr 0.0003 --precision bf16 --num-workers 0 --max-steps 20 --gpus 0,1,2,3'
```

确认三个 variant 都能启动后，正式训练去掉 `MAX_STEPS=20`：

```bash
wintrain '$env:CUDA_VISIBLE_DEVICES="0,1,2,3"; $env:PYTHONUNBUFFERED="1"; conda run --no-capture-output -n yolo_clean python -u D:\ResearchCode\halo-free\Decoupled_Learning\20261030_edgepreserving\exp_a\run_sequence_windows_ddp_gloo.py --variants a1 c1 a0 --data-root D:\ResearchCode\S2R-HDR\data\20260818 --batch-size-per-gpu 8 --crop-size 128 --epochs 200 --lr 0.0003 --precision bf16 --num-workers 0 --gpus 0,1,2,3'
```

修改命令中的 `--batch-size-per-gpu`、`--crop-size`、`--epochs`、`--gpus` 等参数
即可调整训练设置；`--variants a1 c1 a0` 控制实验顺序。串行 launcher 为
`run_sequence_windows_ddp_gloo.py`。

## WinTrain DataParallel

WinTrain 直接调用统一训练器的 `--parallelism dp` 分支，不再维护单独的 DP trainer。
这里 `--batch-size-per-gpu` 会换算成 DataLoader 总 batch；四卡每卡 16 即全局 batch
64。128 patch 下先用 16 做 20 步 smoke test，确认显存后再去掉 `--max-steps 20`。

```bash
wintrain '$env:CUDA_VISIBLE_DEVICES="0,1,2,3"; $env:PYTHONUNBUFFERED="1"; conda run --no-capture-output -n yolo_clean python -u D:\ResearchCode\halo-free\Decoupled_Learning\20261030_edgepreserving\exp_a\src\train_exp_a.py --parallelism dp --variant a1 --data-root D:\ResearchCode\S2R-HDR\data\20260818 --output-dir D:\ResearchCode\halo-free\Decoupled_Learning\20261030_edgepreserving\exp_a\runs\wintrain_dp_a1_patch128_b16 --batch-size-per-gpu 16 --crop-size 128 --epochs 200 --lr 0.0003 --min-lr 0.000003 --lr-schedule cosine --precision bf16 --num-workers 0 --max-steps 20'
```

```bash
wintrain '$env:CUDA_VISIBLE_DEVICES="0,1,2,3"; $env:PYTHONUNBUFFERED="1"; conda run --no-capture-output -n yolo_clean python -u D:\ResearchCode\halo-free\Decoupled_Learning\20261030_edgepreserving\exp_a\src\train_exp_a.py --parallelism dp --variant a0 --data-root D:\ResearchCode\S2R-HDR\data\20260818 --output-dir D:\ResearchCode\halo-free\Decoupled_Learning\20261030_edgepreserving\exp_a\runs\wintrain_dp_a0_patch128_b16 --batch-size-per-gpu 16 --crop-size 128 --epochs 200 --lr 0.0003 --min-lr 0.000003 --lr-schedule cosine --precision bf16 --num-workers 0 --max-steps 20'
```

```bash
wintrain '$env:CUDA_VISIBLE_DEVICES="0,1,2,3"; $env:PYTHONUNBUFFERED="1"; conda run --no-capture-output -n yolo_clean python -u D:\ResearchCode\halo-free\Decoupled_Learning\20261030_edgepreserving\exp_a\src\train_exp_a.py --parallelism dp --variant c1 --data-root D:\ResearchCode\S2R-HDR\data\20260818 --output-dir D:\ResearchCode\halo-free\Decoupled_Learning\20261030_edgepreserving\exp_a\runs\wintrain_dp_c1_patch128_b16 --batch-size-per-gpu 16 --crop-size 128 --epochs 200 --lr 0.0003 --min-lr 0.000003 --lr-schedule cosine --precision bf16 --num-workers 0 --max-steps 20'
```

正式训练时删除每条命令末尾的 `--max-steps 20`。如果每卡显存仍明显低于目标，
可把 `--batch-size-per-gpu 16` 提到 24；不要同时改变 patch 和 batch，先隔离速度
与显存变化。

## 四卡运行

```bash
cd /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261030_edgepreserving
CUDA_VISIBLE_DEVICES=0,1,2,3 VARIANT=a1 MAX_STEPS=20 \
  bash exp_a/run_4gpu_ddp.sh
```

这条命令只做 20 步 NCCL/DDP smoke test。确认日志显示 `world=4` 且成功写出
`latest.pth` 后，正式训练时去掉 `MAX_STEPS`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 VARIANT=a1 EPOCHS=80 \
  bash exp_a/run_4gpu_ddp.sh
```
