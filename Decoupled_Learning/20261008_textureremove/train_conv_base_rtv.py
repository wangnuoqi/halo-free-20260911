"""Stage 1: train ConvGuided base on fixed RTV targets."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from rtv_training_common import DeepGuidedFilter, RTVDataset, base_loss, collate_pairs, load_state, make_low_resolution, seed_everything

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent.parent / "datasets" / "VOC2012" / "RTV_RGB_fixed_lambda_0p01"
MODEL_ROOT = ROOT / "models" / "conv_base_rtv"

def evaluate(model, loader, device):
    model.eval(); total = 0.0
    with torch.no_grad():
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            image = image - 0.5
            total += torch.mean((model(make_low_resolution(image), image) - target) ** 2).item()
    return total / max(1, len(loader))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DATA_ROOT)); parser.add_argument("--model-root", default=str(MODEL_ROOT))
    parser.add_argument("--epochs", type=int, default=80); parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    train_loader = DataLoader(RTVDataset(str(data_root / "train")), args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_pairs)
    val_loader = DataLoader(RTVDataset(str(data_root / "val")), 1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_pairs)
    model = DeepGuidedFilter().to(device)
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr); scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
    save_root = Path(args.model_root); save_root.mkdir(parents=True, exist_ok=True)
    full_path, best_path = save_root / "conv_base_rtv_full.pth", save_root / "conv_base_rtv_best.pth"
    start_epoch, best = 0, float("inf")
    if args.resume and full_path.exists():
        checkpoint = torch.load(full_path, map_location=device); load_state(model, full_path, device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"]); scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch, best = checkpoint["epoch"], checkpoint.get("best_loss", best)
    for epoch in range(start_epoch, args.epochs):
        model.train(); progress = tqdm(train_loader, desc=f"Conv {epoch + 1}/{args.epochs}")
        for image, target in progress:
            image, target = image.to(device), target.to(device)
            image = image - 0.5
            loss = base_loss(model(make_low_resolution(image), image), target)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); progress.set_postfix(loss=f"{loss.item():.5f}")
        val_mse = evaluate(model, val_loader, device); scheduler.step(); print(f"Epoch {epoch + 1}: val MSE={val_mse:.7f}")
        torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "best_loss": min(best, val_mse)}, full_path)
        if val_mse < best: best = val_mse; torch.save(model.state_dict(), best_path); print(f"Saved best base checkpoint: {best_path}")

if __name__ == "__main__": main()
