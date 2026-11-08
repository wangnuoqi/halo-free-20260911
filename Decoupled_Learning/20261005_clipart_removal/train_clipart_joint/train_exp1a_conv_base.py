import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from clipart_conv_base_common import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=r"D:\nuoqi\halo-free\Decoupled_Learning\20261005_clipart_removal\openclipart_5000\jpeg_pairs_512")
    p.add_argument("--list-dir", default=r"D:\nuoqi\halo-free\Decoupled_Learning\20261005_clipart_removal\openclipart_5000")
    p.add_argument("--quality", type=int, default=30)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", default=r"D:\nuoqi\halo-free\Decoupled_Learning\20261005_clipart_removal\train_clipart_joint\ckpt\exp1-a_conv_base")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", default="")
    args = p.parse_args()
    seed_everything(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train = make_dataset(args.data_root, args.quality, "train", args.list_dir, 1)
    val = make_dataset(args.data_root, args.quality, "val", args.list_dir, 1)
    loaders = (DataLoader(train, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True),
               DataLoader(val, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True))
    model = DeepGuidedFilter(1).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
    start, best = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"]); scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start, best = ckpt["epoch"], ckpt.get("best_loss", best)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, args.epochs):
        model.train(); running = 0.0
        for x_lr, x_hr, target in tqdm(loaders[0], desc=f"exp1-a train {epoch+1}/{args.epochs}"):
            x_lr, x_hr, target = x_lr.to(device), x_hr.to(device), target.to(device)
            pred = model(x_lr - 0.5, x_hr - 0.5)
            mse = torch.nn.functional.mse_loss(pred, target)
            grad = sobel_gradient_loss(pred, target)
            loss = mse + grad / (grad / (mse + 1e-8)).detach().clamp_min(1e-8)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); running += mse.item()
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for x_lr, x_hr, target in loaders[1]:
                val_loss += torch.nn.functional.mse_loss(model(x_lr.to(device)-0.5, x_hr.to(device)-0.5), target.to(device)).item()
        val_loss /= len(loaders[1]); scheduler.step()
        print(f"epoch={epoch+1} train_mse={running/len(loaders[0]):.6f} val_mse={val_loss:.6f}")
        save_checkpoint(out / "convguided_exp1a_q30_last.pth", model, optimizer, scheduler, epoch, best, args)
        if val_loss < best:
            best = val_loss; save_checkpoint(out / "convguided_exp1a_q30_best.pth", model, optimizer, scheduler, epoch, best, args)


if __name__ == "__main__":
    main()
