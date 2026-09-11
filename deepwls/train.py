import argparse
import datetime
import os
import random
from pathlib import Path

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss_function.new_loss import new_loss
from net.fcnBn import FcnBn
from util.dataset import PairedImageDataset

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        class SummaryWriter:  # type: ignore[override]
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def add_images(self, *args, **kwargs):
                pass

            def close(self):
                pass


DEFAULT_DATA_ROOT = Path(r"D:\nuoqi\S2R-HDR\data\mixed_lavel_singlehdr")
DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "runs"


def parse_args():
    def parse_patch_size(value):
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"none", "full", "0"}:
            return None
        size = int(text)
        if size <= 0:
            return None
        return size

    parser = argparse.ArgumentParser(description="Train deepwls on paired single-HDR/WLS data.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16, help="Global batch size for DataParallel.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--patch-size",
        type=parse_patch_size,
        default=None,
        help="Training crop size. Use 0/none/full to train on full-resolution images.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lamd", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume-path", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(args):
    train_dataset = PairedImageDataset(
        args.data_root / "traininput",
        args.data_root / "trainlabel",
        split="train",
        patch_size=args.patch_size,
    )
    test_dataset = PairedImageDataset(
        args.data_root / "testinput",
        args.data_root / "testlabel",
        split="test",
        patch_size=None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader


def ensure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("No CUDA devices are visible. Please set CUDA_VISIBLE_DEVICES correctly.")
    return torch.device("cuda"), list(range(device_count))


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def evaluate(model, loader, criterion, device, lamd, alpha):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(inputs, outputs, lamd, alpha, target=targets)
            mse = torch.mean((outputs - targets) ** 2)
            total_loss += loss.item()
            total_mse += mse.item()

    num_batches = max(len(loader), 1)
    return total_loss / num_batches, total_mse / num_batches


def save_checkpoint(save_dir, epoch, model, optimizer, best_metric, is_best):
    save_dir.mkdir(parents=True, exist_ok=True)
    full_path = save_dir / "deepwls_full.pth"
    last_path = save_dir / "deepwls_last.pth"
    best_path = save_dir / "deepwls_best.pth"

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
        },
        full_path,
    )
    torch.save(unwrap_model(model).state_dict(), last_path)
    if is_best:
        torch.save(unwrap_model(model).state_dict(), best_path)


def maybe_resume(args, model, optimizer, device):
    start_epoch = 0
    best_metric = float("inf")
    resume_path = args.resume_path or (args.save_dir / "deepwls_full.pth")

    if args.no_resume or not resume_path.exists():
        print("=> No checkpoint found, training from scratch.")
        return start_epoch, best_metric

    print(f"=> Resuming training from checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    best_metric = checkpoint.get("best_metric", float("inf"))
    print(f"=> Resumed from epoch {start_epoch}, best_metric = {best_metric:.6f}")
    return start_epoch, best_metric


def main():
    args = parse_args()
    set_seed(args.seed)
    device, device_ids = ensure_cuda()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = build_dataloaders(args)

    model = FcnBn()
    model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0]).to(device)
    print(f"=> Using DataParallel device_ids={device_ids}")

    criterion = new_loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch, best_metric = maybe_resume(args, model, optimizer, device)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(args.log_dir / f"deepwls_{timestamp}"))

    epoch_bar = tqdm(range(start_epoch, args.epochs), desc="epochs", dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0

        batch_bar = tqdm(
            train_loader,
            desc=f"train {epoch + 1}/{args.epochs}",
            dynamic_ncols=True,
            leave=False,
        )
        for batch_idx, (inputs, targets, _) in enumerate(batch_bar, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(inputs, outputs, args.lamd, args.alpha, target=targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_bar.set_postfix(loss=f"{running_loss / batch_idx:.6f}")

        avg_train_loss = running_loss / max(len(train_loader), 1)
        avg_val_loss, avg_val_mse = evaluate(model, test_loader, criterion, device, args.lamd, args.alpha)
        is_best = avg_val_mse < best_metric
        if is_best:
            best_metric = avg_val_mse

        save_checkpoint(args.save_dir, epoch, model, optimizer, best_metric, is_best)
        epoch_bar.set_postfix(
            train_loss=f"{avg_train_loss:.6f}",
            val_mse=f"{avg_val_mse:.6f}",
            best=f"{best_metric:.6f}",
        )

        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={avg_train_loss:.6f} "
            f"val_loss={avg_val_loss:.6f} "
            f"val_mse={avg_val_mse:.6f} "
            f"best_val_mse={best_metric:.6f}"
        )

        writer.add_scalar("train/loss", avg_train_loss, epoch + 1)
        writer.add_scalar("val/loss", avg_val_loss, epoch + 1)
        writer.add_scalar("val/mse", avg_val_mse, epoch + 1)
        writer.add_images("train/input", inputs[:4], epoch + 1)
        writer.add_images("train/target", targets[:4], epoch + 1)
        writer.add_images("train/output", outputs[:4], epoch + 1)

    writer.close()


if __name__ == "__main__":
    main()
