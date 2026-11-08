# Single-channel NAFNet baseline for 20260818 WLS

This baseline adapts the completed RTV NAFNet training recipe to the existing
single-channel WLS (`lambda=0.500`, `alpha=1.400`) pairs. It uses the **original
one-channel input directly** (`img_channel=1`, `train_size=(1,1,256,256)`),
not a triplicated RGB input followed by a Y conversion. The GoPro-width-32
NAFNetLocal architecture, PSNR loss, AdamW parameters, cosine schedule, random
paired 256×256 training crops, BF16 forward, three-GPU DDP, and 200,000-update
budget match the earlier RTV NAFNet recipe. A copy of the NAFNet source and its
license is under `baseline/NAFNet/`.

The source data is `/mnt/d/ResearchCode/S2R-HDR/data/20260818/` (override with
`--data-root`). The loader matches `traininput-l0p500-a1p400/*-log.tiff` with
`trainlabel-l0p500-a1p400/*-WLS-l0p500-a1p400.tiff` by source ID and reads the
existing float TIFF values without channel conversion or normalization. The
train split has 2,003 pairs. By default, a deterministic seed-42 selection
reserves 200 of those for full-image validation, leaving 1,803 for optimizer
updates. The separate 200 test pairs are **not loaded during training**. The
split ID hashes and counts are recorded in checkpoint protocol metadata; resume
rejects a changed split or training configuration.

Run from `20260815_edgepreserving/` in Linux or WSL with three CUDA GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONUNBUFFERED=1 \
  /home/nuoqi/miniconda3/envs/halofree/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=3 baseline/train_nafnet_wls.py \
  --data-root /mnt/d/ResearchCode/S2R-HDR/data/20260818 \
  --output-dir baseline/runs/nafnet_wls_patch256_3gpu \
  --total-iter 200000 --batch-size-per-gpu 8 --micro-batch-size 8 \
  --train-crop-size 256 --precision bf16 --num-workers 4
```

For WinTrain on Windows with four CUDA GPUs, use the single-process
`DataParallel` entry point in the `yolo_clean` environment. Run this from the
WSL shell where the `wintrain` function is available:

```bash
wintrain '$env:CUDA_VISIBLE_DEVICES="0,1,2,3"; $env:PYTHONUNBUFFERED="1"; conda run --no-capture-output -n yolo_clean python -u D:\ResearchCode\halo-free\Decoupled_Learning\20260815_edgepreserving\baseline\train_nafnet_wls_wintrain.py --data-root D:\ResearchCode\S2R-HDR\data\20260818 --output-dir D:\ResearchCode\halo-free\Decoupled_Learning\20260815_edgepreserving\baseline\runs\nafnet_wls_patch256_4gpu --batch-size-per-gpu 8 --micro-batch-size 8 --train-crop-size 256 --total-iter 200000 --precision bf16'
```

This uses batch 8 on each card (global batch 32), random paired 256×256
training patches, and full-image validation with odd bottom/right borders
trimmed. The Linux three-GPU DDP entry point remains separate. Add `--resume`
to the Windows command only when continuing its existing latest checkpoint.

For a new WSL/NCCL three-GPU run with batch 32 per GPU (global batch 96), use:

```bash
bash baseline/run_nafnet_wls_nccl_3gpu_bs32.sh
```

This uses the same training entry point as the three-GPU DDP recipe, with a
separate output directory and no resume from the four-GPU Windows run. Each
GPU accumulates two microbatches of 16 into one effective batch of 32; this
keeps memory below the approximately 17.5 GiB peak observed with one
32-image forward on each RTX 3090. The un-split 32-image run stalled under
memory pressure while the existing Windows four-GPU job was also running;
the two-microbatch configuration completed a 30-update NCCL smoke test.
Keep `--total-iter 200000` only if
the intended budget is 200,000 optimizer updates: global batch 96 processes
three times as many training patches as global batch 32 over that budget.
For an approximately equal number of sampled patches, pass
`--total-iter 66667` to the launcher; this changes the cosine schedule and
number of updates, so it is a distinct training protocol.

Outputs are `nafnet_wls_latest.pth` and `nafnet_wls_best.pth` under the output
directory. Add `--resume` to continue from latest with the same data and
protocol. The NAFNet architecture supports full-image inference; validation
uses batch size 1 and removes only an odd bottom row/right column. If memory
allows, `--train-crop-size 0` selects batch-minimum top-left full-image
training, but keep it in a separate output directory because it is a different
training protocol.

The exp7C three-stage checkpoint was selected using its 200-image test set.
This new NAFNet run instead selects on a train-only holdout. For a strictly
independent comparison, evaluate both final models on a new held-out set or
retrain exp7C with a separate validation split. Do not compare training-time
validation numbers from these two different splits as if they were the same
benchmark.

## VIVNet WLS baseline

`vivnet_wls/` adapts the VIVNet architecture used by the Q30 clipart baseline
to the original single-channel WLS input and target. Its stem and output
projection use one channel; the intermediate VIVNet architecture is unchanged.
The copied architecture source and upstream license are kept together in that
folder.

The trainer uses the same seed-42 train/validation partition, random paired
256x256 crops, BF16, and AdamW settings as the NAFNet recipe. It retains
VIVNet's clipart-baseline objective, `L1 + 0.1 * FFT`, so loss is a documented
protocol difference. The recommended run uses three-GPU NCCL/DDP, effective
batch 192 (64 per GPU), accumulated as four microbatches of 16 per GPU, for
100,000 optimizer updates. This processes approximately the same number of
training patches as the four-GPU DataParallel option at batch 256 over 75,000
updates. The learning rate stays at `1e-3`. The launcher defaults to physical
GPUs 1, 2, and 3, and writes to
`baseline/runs/vivnet_wls_patch256_nccl_3gpu_bs64/`.

For a single-GPU run on physical GPU 3, run `train.py` directly with
`--gpu-id 3` and leave `CUDA_VISIBLE_DEVICES` unset. It uses no DDP or
DataParallel wrapper, accumulates
twelve microbatches of 16 for effective batch 192, and writes to
`baseline/runs/vivnet_wls_patch256_1gpu_cuda3_bs192/`. Expect it to run
slower than the three-GPU version.

Do not launch it while NAFNet is occupying GPUs 1-2. After that run finishes,
start VIVNet from the project root with:

```bash
bash baseline/vivnet_wls/run_3gpu_bs64.sh
```

This is a fresh run. The prior three-GPU DDP checkpoint is not resumed because
the parallelism and effective batch differ. Evaluate the independent
200-image test set after training with:

```bash
python baseline/vivnet_wls/evaluate.py \
  --checkpoint baseline/runs/vivnet_wls_patch256_nccl_3gpu_bs64/vivnet_wls_best.pth \
  --output-dir baseline/evaluation/vivnet_wls_test
```

The evaluator saves one prediction per test image, per-image metrics, and a
summary JSON. VIVNet's two downsampling levels require dimensions divisible by
four; inference pads only the bottom/right edges and crops predictions back to
the shared even-cropped input size.
