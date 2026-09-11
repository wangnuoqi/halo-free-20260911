# GDSR / NYUv2 论文清单

## A. 最应该优先对齐的 guided-filter 基线

| 优先级 | 方法 / 论文 | 年份 | NYUv2 关系 | 为什么要看 | 来源 | 证据 |
|---|---|---:|---|---|---|---|
| P0 | Guided Depth Map Super-resolution: A Survey | 2023 | 统一比较 1000/449，x4/x8/x16 | 先确定共同 benchmark 表和协议差异 | [arXiv](https://arxiv.org/abs/2302.09598), [code](https://github.com/zhwzhong/Guided-Depth-Map-Super-resolution-A-Survey) | B |
| P0 | Guided Filter (GF) | 经典基线 | survey NYUv2 表中列为 classical baseline | 用来测量非学习 guided filter 的下限 | [survey table](https://arxiv.org/abs/2302.09598) | B |
| P0 | Deep Guided Filter for High-Resolution Image Processing (DGF) | 2018 | survey NYUv2 表中列为 DGF | 直接对应你说的 guided filter baseline | [project/search](https://arxiv.org/search/?query=Deep+Guided+Filter+for+High-Resolution+Image+Processing&searchtype=all) | B/C |
| P0 | Deep Joint Image Filtering (DJF) | 2016 | GDSR 早期 guided filtering 路线 | 经典 RGB-guided depth filtering 基线 | [survey code](https://github.com/zhwzhong/Guided-Depth-Map-Super-resolution-A-Survey) | B |
| P0 | Joint Image Filtering with Deep Convolutional Networks (DJFR) | 2019 | survey 明确沿用其 first-1000 / remaining-449 约定 | 最重要的历史协议锚点之一 | [survey code](https://github.com/zhwzhong/Guided-Depth-Map-Super-resolution-A-Survey) | B |
| P0 | DKN: Deep Kernel Network for Depth Map Super-Resolution | 2019/2021 | 官方仓库提供 NYUv2 split 和 scale-specific train/test | 你们当前 `.npy` 输出就是为它准备的 | [arXiv](https://arxiv.org/abs/1910.08373), [code](https://github.com/cvlab-yonsei/dkn) | A |
| P0 | FDKN: Fast Depth Map Super-Resolution with Deep Kernel Network | 2021 | 与 DKN 共用公开实现和 NYUv2 任务设置 | guided-kernel baseline；速度/参数量对比 | [code](https://github.com/cvlab-yonsei/dkn) | A/B |

## B. 重要的后续 RGB-guided 方法

| 方法 / 论文 | 年份 | 主要路线 | NYUv2 / 代码来源 | 证据 |
|---|---:|---|---|---|
| ATGV-Net: Accurate Depth Super-Resolution | 2016 | variational + deep | [arXiv](https://arxiv.org/abs/1607.07988) | C |
| Joint Convolutional Neural Pyramid for Depth Map Super-Resolution | 2018 | multi-scale CNN | [arXiv](https://arxiv.org/abs/1801.00968) | C |
| PAG-Net | 2019 | attention-guided CNN | [arXiv](https://arxiv.org/abs/1911.09878) | C |
| Multi-Scale Progressive Fusion Learning | 2020 | multi-scale RGB/depth fusion | [arXiv](https://arxiv.org/abs/2011.11865) | C |
| AHMF: High-resolution Depth Maps Imaging via Attention-based Hierarchical Multi-modal Fusion | 2021/2022 | hierarchical multimodal fusion | [arXiv](https://arxiv.org/abs/2104.01530), [code](https://github.com/zhwzhong/AHMF) | A/B |
| DCTNet: Discrete Cosine Transform Network for Guided Depth Map Super-Resolution | 2022 | DCT + semi-coupled features + edge attention | [arXiv](https://arxiv.org/abs/2104.06977), [code](https://github.com/Zhaozixiang1228/GDSR-DCTNet) | A |
| PAKA / Pixel-Adaptive Kernel Attention | 2022 | content-aware propagation | [arXiv](https://arxiv.org/abs/2107.13144) | C |
| BridgeNet | 2021 | joint DSR and monocular depth estimation | [project](https://rmcong.github.io/proj_BridgeNet.html) | C |
| JIIF: Joint Implicit Image Function for Guided Depth Super-Resolution | 2021 | implicit RGB-depth function | [code](https://github.com/ashawkey/jiif) | B/C |
| FDSR: Towards Fast and Accurate Real-World Depth Super-Resolution | 2021 | real-world benchmark and baseline | [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/He_Towards_Fast_and_Accurate_Real-World_Depth_Super-Resolution_Benchmark_Dataset_and_CVPR_2021_paper.html) | B |
| GraphSR: Learning Graph Regularisation for Guided Super-Resolution | 2022 | graph regularization | [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/de_Lutio_Learning_Graph_Regularisation_for_Guided_Super-Resolution_CVPR_2022_paper.html), [code](https://github.com/prs-eth/graph-super-resolution) | B/C |
| SSDNet: Spherical Space Feature Decomposition for Guided Depth Map Super-Resolution | 2023 | feature-space decomposition | [arXiv](https://arxiv.org/abs/2303.08942), [code](https://github.com/Zhaozixiang1228/GDSR-SSDNet) | C |
| DADA: Depth Super-Resolution from Explicit and Implicit High-Frequency Features | 2023 | explicit/implicit high-frequency modeling | [arXiv](https://arxiv.org/abs/2303.09307) | C |
| SGNet: Structure Guided Network via Gradient-Frequency Awareness | 2023 | gradient/frequency guidance | [arXiv](https://arxiv.org/abs/2312.05799), [code](https://github.com/yanzq95/SGNet) | C |
| DSR-Diff: Guided Depth Super-Resolution by Deep Anisotropic Diffusion | 2023 | diffusion | [arXiv](https://arxiv.org/abs/2211.11592), [code](https://github.com/prs-eth/Diffusion-Super-Resolution) | C |

## C. 经典 / filtering 相关补充

| 方法 / 论文 | 作用 | 来源 |
|---|---|---|
| A Joint Intensity and Depth Co-Sparse Analysis Model | classical RGB-depth joint prior | [arXiv](https://arxiv.org/abs/1304.5319) |
| Guided Deep Decoder: Unsupervised Image Pair Fusion | guided reconstruction, not a direct DSR apples-to-apples baseline | [ECCV](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510086.pdf) |
| Deep edge map guided depth super resolution | edge-guided classical/deep hybrid | [survey index](https://github.com/zhwzhong/Guided-Depth-Map-Super-resolution-A-Survey) |
| Frequency-dependent Depth Map Enhancement | intensity-guided affine/refinement filtering | [code](https://github.com/Yifan-Zuo/Frequency-dependent-Depth-Map-Enhancement) |
| MIG-Net | intensity/gradient-guided depth SR | [code](https://github.com/Yifan-Zuo/MIG-net-gradient_guided_depth_enhancement) |

## 论文数量说明

本清单超过 15 篇，满足 researchA 的初步检索要求。真正用于你们最终 benchmark
的第一批建议只选 P0 的 7 项，再加 DCTNet、AHMF、JIIF；其余作为 related work
或第二批复现对象。
