import argparse
import os
import re
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from baseline.official_model import load_official_model
    from baseline.rtv_data import (
        RTV_LAMBDA,
        RTVPairDataset,
        build_model_input,
        encode_rtv_condition,
        paired_crop_collate,
    )
except ModuleNotFoundError:
    from official_model import load_official_model
    from rtv_data import (
        RTV_LAMBDA,
        RTVPairDataset,
        build_model_input,
        encode_rtv_condition,
        paired_crop_collate,
    )


DEFAULT_DATA_ROOT = r"D:\ResearchCode\halo-free\datasets\VOC2012\RTV_RGB_fixed_lambda_0p01"
DEFAULT_OFFICIAL_SCRIPT = r"D:\ResearchCode\decoupled-learning-fan\DecoupleLearning-master\train_1_operator.py"
DEFAULT_OUTPUT_DIR = r"D:\ResearchCode\halo-free\Decoupled_Learning\20261008_textureremove\baseline\runs\rtv_full_image"


def normalize_path(path):
    match = re.match(r"^([A-Za-z]):\\(.*)$", str(path))
    if os.name == "nt" or not match:
        return Path(path)
    drive, suffix = match.groups()
    return Path(f"/mnt/{drive.lower()}/{suffix.replace(chr(92), '/')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train official DecoupleLearning single-operator model on fixed RTV RGB pairs.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--official-script", default=DEFAULT_OFFICIAL_SCRIPT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--crop-size", type=int, default=None,
        help="Train crop size; default None applies three-stage batch-level full-image collation",
    )
    parser.add_argument(
        "--eval-crop-size", type=int, default=None,
        help="Validation crop size; default None evaluates each complete image",
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--rtv-lambda", type=float, default=RTV_LAMBDA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_run_protocol(args):
    return {
        "train_geometry": "batch_min_top_left_even" if args.crop_size is None else "random_crop",
        "train_crop_size": args.crop_size,
        "validation_geometry": "full_image_even" if args.eval_crop_size is None else "center_crop",
        "validation_crop_size": args.eval_crop_size,
        "batch_size": args.batch_size,
    }


def validate_resume_protocol(checkpoint, current_protocol):
    if checkpoint.get("run_protocol") != current_protocol:
        raise ValueError(
            "Checkpoint training protocol does not match current arguments; "
            "use the matching crop settings and batch size, or a separate output directory"
        )


def evaluate(model, loader, condition_value, device):
    model.eval()
    loss_sum = 0.0
    with torch.no_grad():
        for image, target in loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            model_input = build_model_input(image)
            condition = torch.full((image.shape[0], 1), condition_value, device=device)
            prediction = model([condition, model_input])
            height = min(prediction.shape[-2], target.shape[-2])
            width = min(prediction.shape[-1], target.shape[-1])
            loss_sum += torch.nn.functional.mse_loss(
                prediction[..., :height, :width], target[..., :height, :width]
            ).item()
    return loss_sum / max(1, len(loader))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_root = normalize_path(args.data_root)
    official_script = normalize_path(args.official_script)
    output_dir = normalize_path(args.output_dir)
    if not official_script.is_file():
        raise FileNotFoundError(f"Official DecoupleLearning script not found: {official_script}")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers cannot be negative")
    if args.crop_size is not None and args.crop_size < 2:
        raise ValueError("crop-size must be at least 2 when provided")
    if args.eval_crop_size is not None and args.eval_crop_size < 2:
        raise ValueError("eval-crop-size must be at least 2 when provided")

    train_set = RTVPairDataset(data_root / "train")
    val_set = RTVPairDataset(data_root / "val")
    run_protocol = build_run_protocol(args)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        collate_fn=partial(paired_crop_collate, crop_size=args.crop_size, random_crop=True),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=partial(paired_crop_collate, crop_size=args.eval_crop_size, random_crop=False),
    )
    if len(train_loader) == 0:
        raise RuntimeError("Training set is smaller than one full batch; reduce --batch-size")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = load_official_model(official_script)
    base_model = model_class().to(device)
    condition_value = encode_rtv_condition(args.rtv_lambda)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "rtv_official_latest.pth"
    best_path = output_dir / "rtv_official_best.pth"
    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"No checkpoint to resume: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        validate_resume_protocol(checkpoint, run_protocol)
        base_model.load_state_dict(checkpoint["model_state"])
        start_epoch = checkpoint["epoch"]
        best_val = checkpoint["best_val"]
    model = base_model
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"Resumed from {latest_path}: completed epoch {start_epoch}, next epoch {start_epoch + 1}/{args.epochs}")

    print(f"Official model: {official_script}")
    print(f"RTV lambda={args.rtv_lambda:g}, official condition={condition_value:.6f}")
    print(f"Device={device}, visible GPUs={torch.cuda.device_count()}, batch={args.batch_size}, workers={args.num_workers}")
    train_protocol = "batch-aligned full image" if args.crop_size is None else f"random {args.crop_size}x{args.crop_size} crop"
    val_protocol = "full image" if args.eval_crop_size is None else f"center {args.eval_crop_size}x{args.eval_crop_size} crop"
    print(f"Train pairs={len(train_set)}, validation pairs={len(val_set)}")
    print(f"Training protocol={train_protocol}, validation protocol={val_protocol}, batch={args.batch_size}")

    for epoch in range(start_epoch, args.epochs):
        if epoch == 85:
            for group in optimizer.param_groups:
                group["lr"] = args.lr * 0.1
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for image, target in progress:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            model_input = build_model_input(image)
            condition = torch.full((image.shape[0], 1), condition_value, device=device)
            prediction = model([condition, model_input])
            height = min(prediction.shape[-2], target.shape[-2])
            width = min(prediction.shape[-1], target.shape[-1])
            loss = torch.nn.functional.mse_loss(
                prediction[..., :height, :width], target[..., :height, :width]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, condition_value, device)
        state_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": state_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val": min(best_val, val_loss),
            "rtv_lambda": args.rtv_lambda,
            "rtv_condition": condition_value,
            "run_protocol": run_protocol,
        }
        torch.save(checkpoint, latest_path)
        if val_loss < best_val:
            best_val = val_loss
            checkpoint["best_val"] = best_val
            torch.save(checkpoint, best_path)
        print(f"Epoch {epoch + 1}: train MSE={train_loss:.6f}, val MSE={val_loss:.6f}, best={best_val:.6f}")


if __name__ == "__main__":
    main()
