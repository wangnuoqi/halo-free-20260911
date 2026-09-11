from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import weighted_parametric_cascade_loss

ROOT = Path(__file__).resolve().parent
EXP4 = ROOT.parent / "exp4"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def refinement_inputs(noisy: torch.Tensor, coarse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep Exp6 in the same centered coordinate system as Exp4-A."""
    return noisy - 0.5, coarse - 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp6 loss-only ablation on the unchanged Exp4-A network.")
    parser.add_argument("--name", required=True); parser.add_argument("--final-mse-weight", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=80); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); device = torch.device(args.device)
    data = load_module("exp4_data", EXP4 / "data.py"); model_module = load_module("exp4_model", EXP4 / "model.py")
    import sys
    sys.path.insert(0, str(ROOT.parent / "exp1"))
    from eval_convguided_gray import DeepGuidedFilter, load_model_weights
    train, valid = data.build_datasets(str(ROOT.parents[2] / "data_bsds" / "BSDS 400"), str(ROOT.parents[2] / "data_bsds" / "BSD68"), sigma=25.0)
    stage1 = DeepGuidedFilter().to(device)
    load_model_weights(stage1, str(ROOT.parent / "exp1" / "models" / "convguided_bsds_sigma25" / "convguided_bsds_sigma25_best.pth"), device)
    stage1.eval()
    for p in stage1.parameters(): p.requires_grad_(False)
    model = model_module.DeepParametricCascade3Stage().to(device); optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    out = ROOT / "models" / args.name; out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_train_config.json").write_text(json.dumps(vars(args), indent=2))
    best = float("inf"); history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(DataLoader(train, batch_size=1, shuffle=True), desc=f"{args.name} train {epoch}/{args.epochs}", unit="image")
        for step, batch in enumerate(train_bar, start=1):
            lr, noisy, clean = (batch[k].to(device) for k in ("noisy_lr", "noisy_hr", "clean_hr"))
            with torch.no_grad(): coarse = stage1(lr - .5, noisy - .5)
            refine_noisy, refine_coarse = refinement_inputs(noisy, coarse)
            outputs = model(refine_noisy, refine_coarse); loss, _ = weighted_parametric_cascade_loss(outputs, refine_coarse, clean - .5, final_mse_weight=args.final_mse_weight)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{train_loss / step:.5f}")
        model.eval(); mse = 0.0
        with torch.no_grad():
            valid_bar = tqdm(DataLoader(valid, batch_size=1), desc=f"{args.name} valid {epoch}/{args.epochs}", unit="image", leave=False)
            for batch in valid_bar:
                lr, noisy, clean = (batch[k].to(device) for k in ("noisy_lr", "noisy_hr", "clean_hr"))
                coarse = stage1(lr - .5, noisy - .5)
                refine_noisy, refine_coarse = refinement_inputs(noisy, coarse)
                prediction = torch.clamp(model(refine_noisy, refine_coarse)["final"] + .5, 0, 1)
                mse += torch.mean((prediction - clean).square()).item()
        mse /= len(valid); history.append({"epoch": epoch, "mse": mse})
        (out / "validation_history.json").write_text(json.dumps(history, indent=2))
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict()}, out / "last_full.pth")
        if mse < best: best = mse; torch.save(model.state_dict(), out / "val_best.pth")
        scheduler.step()
        print(f"{args.name} epoch {epoch}/{args.epochs}: BSD68 MSE={mse:.8f}, best={best:.8f}", flush=True)


if __name__ == "__main__":
    main()
