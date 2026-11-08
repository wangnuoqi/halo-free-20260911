"""Evaluate a trained exp1 model on a NAF-compatible cached refine manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import CoarseEdgeGatedRefine3Stage


def main() -> None:
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from refine_cache import CachedRefineDataset, load_cache_entries

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--cache_manifest", default="manifest.jsonl")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.checkpoint, map_location=device)
    config = saved.get("model_config", {})
    model = CoarseEdgeGatedRefine3Stage(config.get("base_channels", 48), config.get("num_blocks", 4), config.get("residual_scale", .20), config.get("gate_beta", .25)).to(device)
    model.load_state_dict(saved.get("model_state_dict", saved), strict=True)
    model.eval()
    dataset = CachedRefineDataset(load_cache_entries(args.cache_root, args.cache_manifest), args.resolution, 1.0, True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    totals, pixels = {"mse": 0.0, "mae": 0.0}, 0
    gate_total, samples = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate", dynamic_ncols=True):
            coarse, blended, prior, gt = (batch[key].to(device) for key in ("prelim", "cond", "prior", "gt"))
            output = model(coarse, blended, prior)
            pred = output["final"]
            totals["mse"] += (pred - gt).square().sum().item()
            totals["mae"] += (pred - gt).abs().sum().item()
            pixels += pred.numel(); gate_total += output["gate3"].mean().item() * pred.shape[0]; samples += pred.shape[0]
    mse, mae = totals["mse"] / pixels, totals["mae"] / pixels
    metrics = {"mse": mse, "mae": mae, "psnr": -10.0 * __import__("math").log10(max(mse, 1e-12)), "mean_stage3_gate": gate_total / samples, "samples": samples}
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
