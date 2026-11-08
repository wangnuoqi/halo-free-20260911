# Single-channel VIVNet for WLS

The VIVNet-T architecture is adapted by changing only its input and output
projections to one channel. The upstream implementation and license are
included in this directory.

Training uses the edge-preserving WLS pairs and the same deterministic
train-only validation holdout as the NAFNet run. The recommended entry point
`train.py` uses three-GPU NCCL/DDP. The loss preserves the VIVNet Clipart baseline objective:
`L1 + 0.1 * FFTLoss`. Checkpoints are written to the separate
run-specific directory.

The recommended large-batch run uses three-GPU NCCL/DDP, random paired
256x256 patches, effective batch size 192 (64 per GPU), BF16, AdamW at
`1e-3`, and 100,000 updates. Four microbatches of 16 are accumulated per GPU.
This processes approximately the same number of training patches as the
four-GPU DataParallel option (global batch 256 over 75,000 updates). The LR
remains `1e-3`; it is not linearly scaled with batch size. The launcher defaults
to physical GPUs 1, 2, and 3, which DDP sees as local devices 0, 1, and 2.

For one-GPU execution on physical GPU 3, run `train.py` directly with
`--gpu-id 3` and without `CUDA_VISIBLE_DEVICES`; this selects the physical
ordinal directly instead of remapping it to logical device 0. Without
`torchrun`, it uses a plain single-GPU model with no DDP or DataParallel
wrapper. The suggested command keeps effective batch 192 by accumulating twelve microbatches of 16 and retains the
100,000-update cosine schedule. Each update computes all 192 samples on one
GPU, so it will run substantially slower than the three-GPU run. Its
checkpoint directory is
`baseline/runs/vivnet_wls_patch256_1gpu_cuda3_bs192/` and it cannot resume
a multi-GPU checkpoint.

Wait for NAFNet to release GPUs 1-2 before starting VIVNet. From the project
root, launch the three-GPU run with:

```bash
bash baseline/vivnet_wls/run_3gpu_bs64.sh
```

The run writes to `baseline/runs/vivnet_wls_patch256_nccl_3gpu_bs64/` and
starts from scratch. It does not resume the four-GPU DataParallel checkpoint
because the parallelism and effective batch differ. Add `--resume` only to
continue a checkpoint from this same three-GPU DDP protocol.

The final independent test-set evaluation command is:

```bash
python baseline/vivnet_wls/evaluate.py \
  --checkpoint baseline/runs/vivnet_wls_patch256_nccl_3gpu_bs64/vivnet_wls_best.pth \
  --output-dir baseline/evaluation/vivnet_wls_test
```
