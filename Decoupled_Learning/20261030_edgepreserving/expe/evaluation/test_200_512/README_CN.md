# Exp E 与原 Exp7C 的 512×512 test-200 对照

2026-11-06 使用 FP32、单卡，在同一 `20260818` test-200 上逐图测量。HR guide/GT 使用中心 512×512 区域，LR 使用与之严格对齐的 256×256 区域；如源图尺寸为奇数，先裁底部/右侧奇数边界。此数据集当前 test triples 本身已是 HR/GT 512×512、LR 256×256，因此中心裁剪不会改变像素区域。每图算 MSE/PSNR，再报告逐图均值；`psnr_from_mean_mse_db` 是另一个聚合口径。

| 模型/输出 | Mean per-image PSNR | Mean image MSE | 说明 |
|---|---:|---:|---|
| 原 Exp7C joint coarse | 33.5392 dB | 4.45145e-4 | 原简单 LRNet + ConvGuided，Joint 后 coarse |
| 原 Exp7C joint final | **47.1141 dB** | 2.27272e-5 | 同一 joint checkpoint 配对原三阶段 Refine |
| Exp E1 Base coarse | 47.1023 dB | 2.27596e-5 | VIVNet LRNet 替换，尚未接外部 Refine |
| Exp E1 + frozen Refine final | 46.7654 dB | 2.36654e-5 | E1 Base 冻结训练 Refine |
| Exp E1 Joint coarse | 47.0932 dB | 2.28519e-5 | Joint best 内的 coarse |
| Exp E1 Joint final | 46.5200 dB | 2.56660e-5 | E1 Joint best 的最终输出 |

逐图结果和复现摘要分别见 `per_image_metrics.csv`、`summary.json`。

复现命令（从 `20261030_edgepreserving` 项目目录执行）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
/home/nuoqi/miniconda3/envs/halofree/bin/python -m expe.evaluate_refine_joint \
  --base-checkpoint expe/runs/scratch/viv_gf_e1/seed42_ddpfix/best.pth \
  --refine-checkpoint expe/runs/three_stage/viv_gf_e1/seed42/refine/refine_best.pth \
  --joint-checkpoint expe/runs/three_stage/viv_gf_e1/seed42/joint/joint_best.pth \
  --legacy-exp7c-conv-checkpoint ../20260815_edgepreserving/experiments/exp7/exp7C_20260818_joint_fidelity/models/exp7C_20260818_joint_fidelity/convguided_joint_best_20260818_l0p500_a1p400.pth \
  --legacy-exp7c-refine-checkpoint ../20260815_edgepreserving/experiments/exp7/exp7C_20260818_joint_fidelity/models/exp7C_20260818_joint_fidelity/rf_3stage_best_20260818_l0p500_a1p400.pth \
  --output-dir expe/evaluation/test_200_512 --device cuda:0 --precision fp32
```

## 训练协议对照

| 设置 | 原 Exp7C Joint | 本次 Exp E1 Refine/Joint | 是否对齐 |
|---|---|---|---|
| 训练数据 | 20260818 WLS | 20260818 WLS | 是 |
| Refine 主体与级联 loss | 原三阶段、0.1/0.3/0.6、Otsu edge loss | 复用原三阶段与对应 loss | 主结构基本对齐 |
| coarse fidelity loss | Joint 加 0.5 × SmoothL1(coarse, GT) | Joint 加同一项 | 是 |
| 训练图像几何 | 完整 512×512 图 | 随机 128×128 HR crop / 64×64 LR | 否 |
| batch | 4 张/step（4卡 DP） | 64 张/step（4卡 DDP，每卡16） | 否 |
| Joint epochs | 80 | 200 | 否 |
| Joint optimizer | Adam | AdamW，weight_decay=1e-4 | 否 |
| Joint 学习率 | Refine 3e-3；Base 3e-4 | Base 与 Refine 同为1e-5 | 否 |
| scheduler | StepLR，每30 epoch ×0.5 | cosine，200 epoch | 否 |
| best 选择 | 原 Exp7C 按 test MSE 选 checkpoint | 内部全图 validation MSE 选 checkpoint | 否，原方案 test-selection 有泄漏风险 |

因此这组测试让推理评估尺寸和数据 split 对齐了，但训练协议没有对齐，**不能将 47.1141 vs 46.5200 dB 解释成仅由简单 LRNet/VIVNet 架构造成的差异**。更具体地说，Exp7C 的 Joint 允许 coarse 大幅退化并依赖 Refine 补偿；E1 Joint 保住了 coarse，但当前 Refine 在 test-200 上反而降低质量。要做架构归因，需要从相同训练样本划分开始，用固定的 512 HR crop、相同 batch/optimizer/scheduler/epoch、相同 Base/Refine/Joint 阶段和仅架构不同的配对训练；checkpoint 只能按 validation 选，再对 test-200 报告。
