# Guided Filtering Comparison Paper Set

This file records the papers that should be treated as the core comparison set for halo-free / guided-filtering experiments.

## Core Comparison Methods

| Tag | Paper | Venue / Year | Method type | Why include in comparison |
| --- | --- | --- | --- | --- |
| DGF | Fast End-to-End Trainable Guided Filter | ECCV 2018 | Deep guided filtering / learnable guided filter layer | Baseline deep guided-filtering method; useful as the neural counterpart of classical guided filtering. |
| WGIF-SK | Weighted Guided Image Filtering With Steering Kernel | IEEE TIP 2019 / TIP vol. 29, 2020 | Classical weighted guided filtering with steering kernel | Directly targets edge preservation and halo artifact reduction by adding orientation-aware local structure. |
| muGIF | Mutually Guided Image Filtering | ACM MM 2017 / IEEE TPAMI 2020 | Classical mutual-structure guided filtering | Handles guidance-target structure inconsistency by preserving mutual structures instead of blindly transferring guidance edges. |
| UMGF | Unsharp Mask Guided Filtering | IEEE TIP 2021 | Classical guided filtering variant with unsharp-mask guidance | Important edge/detail-enhancement guided-filter variant; should be compared for halo/detail tradeoff. |
| DAGF | Deep Attentional Guided Image Filtering | IEEE TNNLS 2023 / vol. 35(9), 2024 | Deep attentional guided filtering | Learns guidance/target dynamic kernels and attention to reduce erroneous structure transfer when guidance and target disagree. |

## Recommended Role in Experiments

- Treat these five methods as the primary comparison group for guided-filtering based halo-free evaluation.
- Split analysis into two families:
  - Classical / model-based: WGIF-SK, muGIF, UMGF.
  - Deep / learnable: DGF, DAGF.
- For qualitative comparison, focus on edge halos, texture leakage, gradient reversal, over-smoothing, and false structure transfer.
- For quantitative comparison, keep PSNR / SSIM if paired ground truth exists, but add local edge-region metrics because halo artifacts are often localized around strong edges.

## Citation Notes

- DGF: Wu et al., Fast End-to-End Trainable Guided Filter, ECCV 2018.
- WGIF-SK: Sun et al., Weighted Guided Image Filtering With Steering Kernel, IEEE Transactions on Image Processing, DOI: 10.1109/TIP.2019.2928631.
- muGIF: Guo et al., Mutually Guided Image Filtering, ACM MM 2017; extended version in IEEE TPAMI, DOI: 10.1109/TPAMI.2018.2883553.
- UMGF: Unsharp Mask Guided Filtering, IEEE Transactions on Image Processing, 2021.
- DAGF: Zhong et al., Deep Attentional Guided Image Filtering, IEEE Transactions on Neural Networks and Learning Systems, DOI: 10.1109/TNNLS.2023.3253472.

## Implementation Tracking

| Tag | Code status | Notes |
| --- | --- | --- |
| DGF | To locate / verify | Search for official or commonly used implementation. |
| WGIF-SK | To locate / implement | Classical method; may need MATLAB/Python reproduction if official code is unavailable. |
| muGIF | To locate / verify | Check author project page / released code availability. |
| UMGF | To locate / implement | Need confirm official code availability and exact parameter settings. |
| DAGF | Code available | Official GitHub exists; verify environment and pretrained models if used. |
