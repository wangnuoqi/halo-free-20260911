"""Exp7C-style joint fine-tuning of a clip-art ConvGuided base and 3-stage Refine."""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from clipart_conv_base_common import DeepGuidedFilter, make_dataset, seed_everything
from clipart_refine_models import CascadeRefine3Stage, cascade_loss


def load_state(path, device):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    return {key.removeprefix("module."): value for key, value in state.items()}


def paired_augment(lr, hr, gt):
    """Apply independent, aligned flips and right-angle rotations per sample."""
    for index in range(hr.shape[0]):
        k = random.randrange(4)
        flip_h = random.random() < 0.5
        flip_v = random.random() < 0.5
        for tensor in (lr, hr, gt):
            if k:
                tensor[index] = torch.rot90(tensor[index], k, dims=(-2, -1))
            if flip_h:
                tensor[index] = tensor[index].flip(-1)
            if flip_v:
                tensor[index] = tensor[index].flip(-2)
    return lr, hr, gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    cli = parser.parse_args()
    config_path = cli.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parent.parent
    script_dir = Path(__file__).resolve().parent

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    channels = int(config["channels"])
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    quality = int(config["quality"])
    data_root = resolve(config["data_root"])
    list_dir = resolve(config["list_dir"])
    base_path = resolve(config["base_checkpoint"])
    refine_path = resolve(config["refine_checkpoint"])
    output_dir = resolve(config["output_dir"])
    patch_size = config.get("train_patch_size")
    if not base_path.is_file() or not refine_path.is_file():
        raise FileNotFoundError(f"Missing warm-start checkpoint: {base_path} or {refine_path}")

    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = torch.device(config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    base = DeepGuidedFilter(channels)
    base.load_state_dict(load_state(base_path, "cpu"), strict=True)
    refine = CascadeRefine3Stage(channels=channels)
    refine.load_state_dict(load_state(refine_path, "cpu"), strict=True)
    base.requires_grad_(True)
    base, refine = base.to(device), refine.to(device)

    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    batch_size = int(config["batch_size"])
    if gpu_count > 1:
        if batch_size < gpu_count:
            raise ValueError("batch_size must be at least the visible GPU count")
        base = torch.nn.DataParallel(base)
        refine = torch.nn.DataParallel(refine)

    base_params = base.parameters()
    refine_params = refine.parameters()
    optimizer = torch.optim.Adam([
        {"params": refine_params, "lr": float(config["refine_lr"])},
        {"params": base_params, "lr": float(config["base_lr"])},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(config["scheduler_step_size"]),
        gamma=float(config["scheduler_gamma"]),
    )
    start_epoch, best_mse = 0, float("inf")
    if cli.resume:
        saved = torch.load(cli.resume, map_location=device)
        saved_config = saved.get("config", {})
        if saved_config.get("train_patch_size") != patch_size or saved_config.get("augment", True) != config.get("augment", True):
            raise ValueError("Resume training geometry/augmentation mismatch; start a new joint run from base/refine checkpoints")
        base.module.load_state_dict(saved["base_state_dict"]) if isinstance(base, torch.nn.DataParallel) else base.load_state_dict(saved["base_state_dict"])
        refine.module.load_state_dict(saved["refine_state_dict"]) if isinstance(refine, torch.nn.DataParallel) else refine.load_state_dict(saved["refine_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_epoch, best_mse = int(saved["epoch"]), float(saved["best_val_mse"])

    loaders = {}
    for split in ("train", "val"):
        dataset = make_dataset(data_root, quality, split, list_dir, channels, patch_size=patch_size)
        loaders[split] = DataLoader(
            dataset, batch_size=(batch_size if split == "train" else int(config.get("val_batch_size", batch_size))), shuffle=(split == "train"),
            num_workers=int(config["num_workers"]), pin_memory=(device.type == "cuda"),
            persistent_workers=int(config["num_workers"]) > 0,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    base_raw = base.module if isinstance(base, torch.nn.DataParallel) else base
    refine_raw = refine.module if isinstance(refine, torch.nn.DataParallel) else refine
    history_path = output_dir / "history.jsonl"
    epochs = int(config["epochs"])
    print(f"Joint exp7C warm-start: channels={channels}, GPUs={gpu_count}, batch={batch_size}, "
          f"base_lr={config['base_lr']}, refine_lr={config['refine_lr']}, "
          f"train_patch={patch_size}, augment={config.get('augment', True)}, validation=full_image", flush=True)

    for epoch in range(start_epoch, epochs):
        base.train()
        refine.train()
        loss_sum, mse_sum, seen = 0.0, 0.0, 0
        progress = tqdm(loaders["train"], desc=f"joint train {epoch + 1}/{epochs}", mininterval=1.0)
        for lr, hr, gt in progress:
            lr, hr, gt = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            if config.get("augment", True):
                lr, hr, gt = paired_augment(lr, hr, gt)

            coarse = base(lr - 0.5, hr - 0.5)
            raw_input, coarse_centered, target = hr - 0.5, coarse - 0.5, gt - 0.5
            outputs = refine(raw_input, coarse_centered)
            loss, stage_losses = cascade_loss(
                outputs, coarse_centered, target,
                thresh_mode="otsu", dilate=0, loss_type="l1", normalize_grad=True,
            )
            loss = loss + float(config["coarse_fidelity_weight"]) * F.smooth_l1_loss(coarse_centered, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            count = gt.shape[0]
            final = (outputs["final"].detach() + 0.5).clamp(0, 1)
            mse = F.mse_loss(final, gt).item()
            loss_sum += loss.item() * count
            mse_sum += mse * count
            seen += count
            progress.set_postfix(loss=f"{loss.item():.5f}", mse=f"{mse:.6f}",
                                 base_lr=f"{optimizer.param_groups[1]['lr']:.1e}")

        scheduler.step()
        base.eval()
        refine.eval()
        val_sse, val_count, val_base_sse = 0.0, 0, 0.0
        with torch.no_grad():
            for lr, hr, gt in tqdm(loaders["val"], desc=f"joint val {epoch + 1}/{epochs}", mininterval=1.0):
                lr, hr, gt = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True), gt.to(device, non_blocking=True)
                coarse = base(lr - 0.5, hr - 0.5)
                output = (refine(hr - 0.5, coarse - 0.5)["final"] + 0.5).clamp(0, 1)
                val_sse += (output - gt).square().sum().item()
                val_base_sse += (coarse - gt).square().sum().item()
                val_count += gt.numel()
        val_mse = val_sse / val_count
        best_mse = min(best_mse, val_mse)
        record = {"epoch": epoch + 1, "train_loss": loss_sum / seen,
                  "train_final_mse": mse_sum / seen, "val_mse": val_mse,
                  "val_base_mse": val_base_sse / val_count,
                  "refine_lr": optimizer.param_groups[0]["lr"],
                  "base_lr": optimizer.param_groups[1]["lr"]}
        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        saved = {"epoch": epoch + 1, "best_val_mse": best_mse,
                 "base_state_dict": base_raw.state_dict(), "refine_state_dict": refine_raw.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                 "config": config}
        torch.save(saved, output_dir / "joint_last.pth")
        if val_mse <= best_mse:
            torch.save(saved, output_dir / "joint_best.pth")
        print(record, flush=True)


if __name__ == "__main__":
    main()
