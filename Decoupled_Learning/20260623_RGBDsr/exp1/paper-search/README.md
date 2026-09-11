# RGB-D Depth Super-Resolution Paper Search

本目录用于核对 RGB-guided depth super-resolution (GDSR) 论文在 NYUv2
benchmark 上的训练和评测协议，特别关注 guided filter / guided filtering
路线。当前材料按 2026-08-02 的检索结果整理。

## 先看什么

1. [`paper_list.md`](paper_list.md)：论文、代码、相关性和证据等级。
2. [`nyuv2_protocol_matrix.md`](nyuv2_protocol_matrix.md)：各方法的 NYUv2
   划分、降采样、训练方式和指标。
3. [`alignment_plan.md`](alignment_plan.md)：把当前工程对齐到文献 benchmark
   的具体建议。

## 当前结论

- 论文中最常见的 NYUv2 GDSR benchmark 约定是：前 1000 对 RGB-D 用于训练，
  后 449 对用于测试。
- NYUv2 的 HR depth 通常通过 bicubic 生成 x4/x8/x16 的 LR depth；但不同代码
  对归一化、裁剪、深度单位和测试图像预处理并不完全一致。
- GF 是 classical guided-filter baseline；DGF 是 Deep Guided Filter；DJF/DJFR
  是 guided filtering 发展链；DKN/FDKN 是后续公开的深度引导网络。
- 不能直接把不同来源的 RMSE 放在同一排名里。必须同时确认 split、scale、LR
  degradation、depth unit、normalization、crop/mask 和 metric。

## 证据等级

- **A**：论文原文或官方代码/README 明确给出。
- **B**：官方 survey 的统一实验或官方仓库间接给出。
- **C**：只有二手资料、搜索摘要或尚未核对原始全文；不能作为最终协议证据。

本轮先完成可核对的清单和协议框架；需要最终复现某个方法时，再逐篇下载 PDF、
读取训练代码和测试代码。
