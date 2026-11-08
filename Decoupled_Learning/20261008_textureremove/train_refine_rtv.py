"""Stage 2: freeze ConvGuided base and train the 3-stage refine network."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from rtv_training_common import CascadeRefine3Stage, DeepGuidedFilter, RTVDataset, cascade_loss, collate_pairs, load_state, make_low_resolution, seed_everything

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent.parent / "datasets" / "VOC2012" / "RTV_RGB_fixed_lambda_0p01"
BASE_PATH = ROOT / "models" / "conv_base_rtv" / "conv_base_rtv_best.pth"
MODEL_ROOT = ROOT / "models" / "refine_rtv"

def evaluate(model, base, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            image = image - 0.5
            coarse = base(make_low_resolution(image), image)
            total += torch.mean((model(image, coarse)["final"].clamp(0, 1) - target) ** 2).item()
    return total / max(1, len(loader))


def restore_refine_checkpoint(model, optimizer, scheduler, full_path, best_path, device):
    """Resume exactly from a full checkpoint, or warm-start from legacy best weights."""
    if full_path.exists():
        checkpoint = torch.load(full_path, map_location=device)
        load_state(model, full_path, device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Resumed full refine checkpoint at epoch {checkpoint['epoch']}: {full_path}")
        return checkpoint["epoch"], checkpoint.get("best_loss", float("inf")), False
    if best_path.exists():
        load_state(model, best_path, device)
        print(f"Loaded legacy best refine weights: {best_path}")
        return 0, float("inf"), True
    raise FileNotFoundError(f"No refine checkpoint found at {full_path} or {best_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DATA_ROOT)); parser.add_argument("--base-checkpoint", default=str(BASE_PATH)); parser.add_argument("--model-root", default=str(MODEL_ROOT))
    parser.add_argument("--epochs", type=int, default=80); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--lr", type=float, default=1e-3); parser.add_argument("--num-workers", type=int, default=8); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume full checkpoint, or continue from legacy best weights")
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    train_loader = DataLoader(RTVDataset(str(data_root / "train")), args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_pairs)
    val_loader = DataLoader(RTVDataset(str(data_root / "val")), 1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_pairs)
    base = load_state(DeepGuidedFilter().to(device), args.base_checkpoint, device).eval()
    for parameter in base.parameters(): parameter.requires_grad_(False)
    model = CascadeRefine3Stage().to(device)
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr); scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
    save_root = Path(args.model_root); save_root.mkdir(parents=True, exist_ok=True)
    full_path, best_path = save_root / "refine_rtv_full.pth", save_root / "refine_rtv_best.pth"
    start_epoch, best = 0, float("inf")
    if args.resume:
        start_epoch, best, resumed_from_best = restore_refine_checkpoint(model, optimizer, scheduler, full_path, best_path, device)
        if resumed_from_best:
            best = evaluate(model, base, val_loader, device)
            print(f"Legacy best val MSE={best:.7f}; optimizer and scheduler start fresh; --epochs counts additional epochs")
    for epoch in range(start_epoch, args.epochs):
        model.train(); progress = tqdm(train_loader, desc=f"Refine {epoch + 1}/{args.epochs}")
        for image, target in progress:
            image, target = image.to(device), target.to(device)
            image = image - 0.5
            with torch.no_grad(): coarse = base(make_low_resolution(image), image)
            loss = cascade_loss(model(image, coarse), coarse, target); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); progress.set_postfix(loss=f"{loss.item():.5f}")
        scheduler.step(); val_mse = evaluate(model, base, val_loader, device)
        print(f"Epoch {epoch + 1}: val MSE={val_mse:.7f}")
        if val_mse < best:
            best = val_mse
            torch.save(model.state_dict(), best_path)
            print(f"Saved best refine checkpoint: {best_path}")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": best,
        }, full_path)

if __name__ == "__main__": main()
