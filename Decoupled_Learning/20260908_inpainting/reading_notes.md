# CRA/HiFill 与 ProFill 精读摘要

## CRA / HiFill（CVPR 2020）

### 任务
超高分辨率图像修复（image inpainting），包括物体/行人删除。

### 逻辑
深度修复网络只在 512×512 低分辨率上运行，负责生成被 mask 区域的粗略、语义合理内容。高分辨率输入先被缩小再放大，形成 blurry image。原始高分辨率图像减去该 blurry image 得到高频 residual。网络用低分辨率特征计算 mask 内外 patch 的相似度，再用这些权重从 mask 外的高频 residual patch 中加权聚合出 mask 内 residual。最后把聚合 residual 加到上采样的低分辨率修复结果上；mask 外直接复制原图。

### 核心公式
`R_H = U(R_L) + Delta_H^agg`（mask 内）；mask 外为 `I_H`。

### 为什么这样做
低分辨率网络擅长大范围结构和语义，但直接放大会模糊；原图外部区域仍保留真实高频纹理，因此从上下文借 residual 可以恢复清晰细节，同时避免 4K 网络的显存开销。

### 与本项目的差异
CRA 的 residual 是 `I_H - U(D(I_H))` 的高频纹理 residual，不是低分辨率原图与低分辨率消除结果的差；其上采样/填充算子是 contextual attention + residual aggregation，不是 WLS。

## ProFill（ECCV 2020）

### 任务
高分辨率图像修复和真实物体删除。

### 逻辑
先在低分辨率上运行 coarse-to-fine GAN inpainting。生成器同时输出修复结果和 confidence map。每轮只保留高置信度生成像素，把低置信度区域继续作为 hole 重修，直到结果稳定。随后用 guided upsampling 将低分辨率结果提升到高分辨率：从原始高分辨率输入的有效区域提取 feature patches，依据低分辨率结果计算的 patch 相似度进行加权借用，再由浅层重建网络生成高分辨率图像。

### 为什么这样做
大洞修复存在歧义，一次生成可能只有局部正确；confidence feedback 让模型逐轮保留可靠部分、重做不可靠部分。低分辨率负责语义，高分辨率 patch 负责细节，因此降低计算量并减少放大模糊。

### 与本项目的差异
ProFill 不显式计算 `I_L - R_L` 并上采样，也不从 4K 原图减 residual；它使用学习式 patch matching/guided upsampling。它和本项目共享“低分辨率语义消除 + 高分辨率细化”的大框架。

## 最简单的对比

两篇论文都在做：小图负责“想清楚要填什么”，大图负责“把结果变清晰”。

- CRA/HiFill：借用高分辨率高频 residual。
- ProFill：借用高分辨率 feature patch，并用 confidence 反复修复。
- 本项目：希望用 WLS 将低分辨率消除变化量恢复到高分辨率，再与原图重建。
