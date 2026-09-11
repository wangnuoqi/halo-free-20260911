# BSDS400 -> BSD68 Grayscale Denoising Performance Report

Scope: grayscale AWGN image denoising. The target protocol is training on BSD/BSDS Train400 (often written as BSD400, 400 Berkeley images, or 400 images of size 180x180 from BSD) and testing on BSD68 / Set68 grayscale images. The common noise levels are sigma = 15, 25, and 50; some papers also report sigma = 5.

## Key Takeaways

- The classic reference point is DnCNN-S: 31.73 / 29.23 / 26.23 dB on BSD68 at sigma = 15 / 25 / 50.
- NLRN is a strong BSD400-trained recurrent/non-local baseline: 31.88 / 29.41 / 26.47 dB.
- SASC-CNN-SS reports very high BSD68 results under Train400 training: 32.03 / 29.63 / 26.66 dB, but it is heavier/iterative and should be checked carefully if used as a fair runtime baseline.
- FFDNet is often shown in BSD68 comparison tables, but its original training protocol commonly uses a larger mixed dataset. Treat it as a useful benchmark, but do not silently mix it into a strict BSD400-only comparison unless using a reproduced BSD400-trained model.

## Strict / Near-Strict BSD400 -> BSD68 Results

Values are average PSNR in dB unless SSIM is explicitly shown. `-` means not reported in the source table.

| Method | Paper / Article | Train setting noted | Test | sigma=5 | sigma=15 | sigma=25 | sigma=50 | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| BM3D | Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering, TIP 2007 | No external training | BSD68 gray | 37.57 | 31.07 | 28.57 | 25.62 | Classical non-learning anchor. |
| LSSC | Learned Simultaneous Sparse Coding / non-local sparse coding baseline | No deep training / model-based | BSD68 gray | 37.70 | 31.28 | 28.71 | 25.72 | Often included as sparse prior baseline. |
| TNRD | Trainable Nonlinear Reaction Diffusion, PAMI 2017 | Train400 / BSD-derived training protocol | BSD68 gray | - | 31.42 | 28.92 | 25.96-25.97 | Important pre-CNN trainable diffusion baseline. |
| CSCnet | Deep convolutional sparse coding network | Report table says exception to BSD400 training in one source | BSD68 gray | 37.84 | 31.57 | 29.11 | 26.24 | Include with caveat; training split differs in some reports. |
| LKSVD | Learned K-SVD style baseline | BSD400-style comparison table | BSD68 gray | - | 31.54 | 29.07 | 26.13 | Lightweight learned sparse baseline. |
| DnCNN-S | Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising, TIP 2017 | Train400; noise-specific models | BSD68 gray | 37.68 | 31.73 | 29.22-29.23 | 26.23 | Canonical CNN baseline for this protocol. |
| DnCNN-B | Same DnCNN paper, blind Gaussian model | Train400; blind sigma in a range | BSD68 gray | - | 31.61 | 29.16 | 26.23 | Slightly lower at low/mid sigma, similar at sigma=50. |
| FFDNet | FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising, TIP 2018 | Often larger mixed training set; appears in BSD400 comparison tables | BSD68 gray | - | 31.63 | 29.19 | 26.29 | Useful flexible/blind baseline; mark training protocol carefully. |
| NLRN | Non-Local Recurrent Network for Image Restoration, NeurIPS 2018 | BSD400 in non-local sparse model comparison supplement | BSD68 gray | 37.92 | 31.88 | 29.41 | 26.47 | Strong non-local learned baseline. |
| SC baseline | Fully Trainable and Interpretable Non-Local Sparse Models, ECCV 2020 | BSD400 | BSD68 gray | 37.84 | 31.46 | 28.90 | 25.84 | Interpretable sparse coding baseline. |
| GroupSC | Fully Trainable and Interpretable Non-Local Sparse Models, ECCV 2020 | BSD400 | BSD68 gray | 37.95 | 31.71 | 29.20 | 26.17 | Lightweight/interpretable; below NLRN but parameter-efficient. |
| SASC-CNN-SS | Learning Hybrid Sparsity Prior for Image Restoration | train400 patches; BSD68 benchmark | BSD68 gray | - | 32.03 / 0.8870 | 29.63 / 0.8289 | 26.66 / 0.7254 | Highest among the located Train400-style reports; verify runtime and implementation availability. |
| ELU+TV | An ELU Network with Total Variation for Image Denoising | BSD/BSD68 non-overlap; exact Train400 split needs verify | BSD68 gray | - | 31.82 | 29.34 | 26.32 | Useful older CNN variant; not as cleanly documented as DnCNN Train400. |

## Extra BSD68 Gray Results Often Quoted, But Not Strict BSD400-Only

These are useful as upper/modern references, but their training data is larger or different. Do not use them as a strict apples-to-apples BSD400-only comparison.

| Method | Paper / Article | Training note | BSD68 sigma=15 | BSD68 sigma=25 | BSD68 sigma=50 | Why keep it |
| --- | --- | --- | ---: | ---: | ---: | --- |
| DRUNet | Plug-and-Play Image Restoration with Deep Denoiser Prior / DPIR | Larger denoiser-prior training | 31.91 | 29.48 | 26.59 | Strong modern CNN/UNet prior baseline. |
| SwinIR | SwinIR: Image Restoration Using Swin Transformer | Larger mixed training data | 31.97 | 29.50 | 26.58 | Transformer baseline. |
| Restormer | Restormer: Efficient Transformer for High-Resolution Image Restoration | Larger mixed training data | 31.96 | 29.52 | 26.62 | Strong transformer restoration baseline. |
| SCUNet | Practical Blind Image Denoising via Swin-Conv-UNet and Data Synthesis | Waterloo + DIV2K + Flickr2K for synthetic Gaussian section | 31.99 | 29.55 | 26.67 | Useful modern ceiling, but not BSD400-only. |

## Suggested Comparison Set for Our Experiments

For a fair BSD400-trained grayscale denoising comparison:

1. BM3D: classical non-learning anchor.
2. TNRD: trainable diffusion / pre-CNN learned baseline.
3. DnCNN-S and/or DnCNN-B: canonical CNN baseline.
4. FFDNet: include only if we clearly label training data; otherwise place in non-strict comparison.
5. NLRN: strong non-local learned baseline.
6. SASC-CNN-SS or GroupSC: interpretable sparse/non-local learned baseline, depending on code availability and runtime budget.

For reporting, use sigma = 15, 25, 50 first. Add sigma = 5 only if the target paper/table includes it.

## Source Notes

- DnCNN official repository reports BSD68 PSNR for BM3D, WNNM, EPLL, MLP, CSF, TNRD, DnCNN, DnCNN-B, FDnCNN and DRUNet: https://github.com/cszn/DnCNN
- DnCNN Train400 issue notes that Train400 data from BSD was used following TNRD / shrinkage-fields style protocols: https://github.com/cszn/DnCNN/issues/42
- FFDNet official repository and paper page: https://github.com/cszn/FFDNet and https://arxiv.org/abs/1710.04026
- SCUNet arXiv HTML Table 1 reports BSD68 gray PSNR for DnCNN, FFDNet, N3Net, NLRN, FOCNet, DAGL, DRUNet, SwinIR, Restormer, SCUNet: https://arxiv.org/html/2203.13278v4
- Fully Trainable and Interpretable Non-Local Sparse Models supplementary material reports BSD68 grayscale denoising with BSD400 training and PSNR/SSIM for BM3D, DnCNN, NLRN, GroupSC: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670239-supp.pdf
- Learning Hybrid Sparsity Prior for Image Restoration reports train400 training patches and BSD68 PSNR/SSIM for SASC variants: https://scispace.com/pdf/learning-hybrid-sparsity-prior-for-image-restoration-where-1sv6ly7b3r.pdf
- An ELU Network with Total Variation for Image Denoising reports BSD68 gray PSNR table: https://arxiv.org/abs/1708.04317 and mirror text at https://www.academia.edu/34386624/An_ELU_Network_with_Total_Variation_for_Image_Denoising

## Open Checks

- Verify whether we want strict `BSD400 only` or allow the common `larger mixed training set but tested on BSD68` papers in the same table.
- Locate official/reliable code for SASC-CNN-SS and GroupSC before committing them as runnable baselines.
- If FFDNet is used, explicitly record whether we run the original released model or retrain/reproduce on BSD400.
