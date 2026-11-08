"""Stage 3: jointly fine-tune the ConvGuided base and refine network."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from rtv_training_common import CascadeRefine3Stage, DeepGuidedFilter, RTVDataset, base_loss, cascade_loss, collate_pairs, load_state, make_low_resolution, seed_everything

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent.parent / "datasets" / "VOC2012" / "RTV_RGB_fixed_lambda_0p01"
BASE_PATH = ROOT / "models" / "conv_base_rtv" / "conv_base_rtv_best.pth"
REFINE_PATH = ROOT / "models" / "refine_rtv" / "refine_rtv_best.pth"
MODEL_ROOT = ROOT / "models" / "joint_rtv"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DATA_ROOT)); parser.add_argument("--base-checkpoint", default=str(BASE_PATH)); parser.add_argument("--refine-checkpoint", default=str(REFINE_PATH)); parser.add_argument("--model-root", default=str(MODEL_ROOT))
    parser.add_argument("--epochs", type=int, default=80); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--base-lr", type=float, default=1e-5); parser.add_argument("--refine-lr", type=float, default=3e-4); parser.add_argument("--num-workers", type=int, default=8); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    train_loader = DataLoader(RTVDataset(str(data_root / "train")), args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_pairs)
    val_loader = DataLoader(RTVDataset(str(data_root / "val")), 1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_pairs)
    base = load_state(DeepGuidedFilter().to(device), args.base_checkpoint, device); refine = load_state(CascadeRefine3Stage().to(device), args.refine_checkpoint, device)
    if torch.cuda.device_count() > 1: base, refine = torch.nn.DataParallel(base), torch.nn.DataParallel(refine)
    optimizer = torch.optim.Adam([{"params": base.parameters(), "lr": args.base_lr}, {"params": refine.parameters(), "lr": args.refine_lr}]); scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
    save_root = Path(args.model_root); save_root.mkdir(parents=True, exist_ok=True); best_path = save_root / "joint_rtv_best.pth"; best = float("inf")
    for epoch in range(args.epochs):
        base.train(); refine.train(); progress = tqdm(train_loader, desc=f"Joint {epoch + 1}/{args.epochs}")
        for image, target in progress:
            image, target = image.to(device), target.to(device)
            image = image - 0.5
            coarse = base(make_low_resolution(image), image); outputs = refine(image, coarse); loss = cascade_loss(outputs, coarse, target) + 0.5 * base_loss(coarse, target)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); progress.set_postfix(loss=f"{loss.item():.5f}")
        scheduler.step(); base.eval(); refine.eval(); total = 0.0
        with torch.no_grad():
            for image, target in val_loader:
                image, target = image.to(device), target.to(device)
                image = image - 0.5
                output = refine(image, base(make_low_resolution(image), image))["final"].clamp(0, 1); total += torch.mean((output - target) ** 2).item()
        val_mse = total / max(1, len(val_loader)); print(f"Epoch {epoch + 1}: val MSE={val_mse:.7f}")
        if val_mse < best: best = val_mse; torch.save({"base": base.state_dict(), "refine": refine.state_dict()}, best_path); print(f"Saved best joint checkpoint: {best_path}")

if __name__ == "__main__": main()
