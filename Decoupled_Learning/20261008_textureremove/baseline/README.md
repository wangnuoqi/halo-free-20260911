# Official DecoupleLearning RTV Baseline

This baseline uses the `Model` and `Resize` class definitions from the official
`DecoupleLearning-master/train_1_operator.py` without importing its old
path-bound training script. The source remains in the separate official
checkout and is loaded at runtime with `--official-script`.

The adapted data flow is:

```text
RTV RGB TIFF input [0, 1]
  -> scale to [0, 255], center RGB by 128, append official edge channel
  -> official single-operator Model with RTV condition 0.035
  -> MSE against paired RTV RGB TIFF label scaled to [0, 255]
```

The condition value follows the RTV conversion in the official training
script: `(lambda - 0.002) * 4.125 + 0.002`. The current data uses `lambda=0.01`,
so the condition is `0.035`. Sigma, sharpness, and iteration count are fixed in
the data generator and are not model inputs in the official single-operator
network.

Training now matches the three-stage loader: `--batch-size 8`, shuffle enabled,
and no dropped final batch. Each training batch is cropped from the top-left to
the smallest height/width in that batch, then an odd bottom row/right column is
trimmed. Validation uses batch size 1 and each complete image. New checkpoints
are written under `runs/rtv_full_image/`, separate from the historical
256x256-crop run in `runs/rtv/`. This keeps the old crop-trained result intact
and prevents accidental resume from that checkpoint. Checkpoints record the
train geometry, validation geometry, crop sizes, and batch size; `--resume`
rejects a checkpoint if those settings differ from the current command.

The official model's input encoding and MSE objective remain unchanged. Its
three color planes use the same BGR order as the three-stage loader; it also
receives the official edge channel, uses RTV condition `0.035`,
and predicts in `[0,255]`; the three-stage model has a different architecture,
input encoding, and loss. The shared protocol is the RGB RTV data,
train/validation/test split, batch size, batch-level image sizing, and
full-image validation. The architecture and objective remain different by
design.

To reproduce the historical crop-trained baseline, pass an explicit crop size
and a separate output directory. Set both crop arguments to use crop validation
as well:

```powershell
--crop-size 256 --eval-crop-size 256 --batch-size 8 --output-dir D:\ResearchCode\halo-free\Decoupled_Learning\20261008_textureremove\baseline\runs\rtv_crop
```

Run from this WSL/Linux environment with Python directly:

```bash
cd /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261008_textureremove
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 /home/nuoqi/miniconda3/envs/halofree/bin/python -u baseline/train_rtv.py \
  --data-root /mnt/d/ResearchCode/halo-free/datasets/VOC2012/RTV_RGB_fixed_lambda_0p01 \
  --official-script /mnt/d/ResearchCode/decoupled-learning-fan/DecoupleLearning-master/train_1_operator.py \
  --output-dir /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261008_textureremove/baseline/runs/rtv_full_image \
  --epochs 100 --batch-size 8 --num-workers 8
```

To resume an interrupted full-image run from
`runs/rtv_full_image/rtv_official_latest.pth`, run the same command with
`--resume` appended. The checkpoint restores the model,
Adam optimizer, completed epoch, and best validation MSE. `--epochs 100` is
the total target epoch count, not 100 additional epochs. Only completed epochs
are saved; any interrupted epoch is run again from its beginning.

The official training script has import-time data and training side effects,
so the baseline extracts only its two model class definitions from the source
AST. No files in the official checkout are modified.

## NAFNet RTV patch-training baseline

`train_nafnet_rtv.py` trains the vendored NAFNet GoPro width-32 architecture
with the official optimizer, loss, and iteration schedule, using our paired
RGB RTV TIFFs. Input and target use the existing loader's BGR order and `[0,1]`
range. Training randomly crops a paired `256x256` patch from each image; the
same crop coordinates are used for input and target. Images smaller than the
patch are reflected along the bottom/right edges before cropping, following
BasicSR's paired-image padding behavior. Validation and inference use one
complete image at a time. NAFNet pads internally to a multiple of 16 and
returns the original image size.

The default recipe is AdamW (`lr=1e-3`, `weight_decay=1e-3`, betas `[0.9,0.9]`),
official PSNR loss, 200,000 optimizer updates, and cosine annealing to `1e-7`.
Three-GPU DDP uses batch size 8 per GPU (effective batch 24) and processes the
full per-GPU batch in one forward/backward pass by default. BF16 autocast is
used for NAFNet forwards; PSNR loss and reported metrics are computed in FP32.
Pinned host memory is disabled because full-image prefetching triggered CUDA
OOMs in the pin-memory worker. With this dataset and setup, one epoch is about
385 updates, so 200,000 updates are approximately 520 epochs. Each update now
processes 24 patches instead of 24 batch-aligned full images, reducing the
pixel workload while retaining the official 200,000-update schedule.
The three-GPU run has fewer samples per update than the official eight-GPU
GoPro configuration; the optimizer update budget and per-GPU batch remain as
configured in the official recipe. Checkpoints are isolated under
`runs/nafnet_rtv_patch256_3gpu/`, separate from the full-image training run.

Run in Linux or WSL with NCCL-enabled PyTorch and all three GPUs visible:

```bash
cd /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261008_textureremove
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONUNBUFFERED=1 /home/nuoqi/miniconda3/envs/halofree/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=3 baseline/train_nafnet_rtv.py \
  --data-root /mnt/d/ResearchCode/halo-free/datasets/VOC2012/RTV_RGB_fixed_lambda_0p01 \
  --output-dir /mnt/d/ResearchCode/halo-free/Decoupled_Learning/20261008_textureremove/baseline/runs/nafnet_rtv_patch256_3gpu \
  --total-iter 200000 --batch-size-per-gpu 8 --micro-batch-size 8 --train-crop-size 256 --precision bf16 --num-workers 4
```

Append `--resume` to restore the latest model, optimizer, scheduler, and
iteration position. Resume checks the recorded data geometry, GPU count,
effective batch, crop geometry, and training settings; the micro-batch size may
change to tune memory use. Use `--train-crop-size 0` to select the earlier
batch-aligned full-image training protocol and a separate output directory.
NCCL requires Linux/WSL; native Windows PyTorch does not provide this backend.
