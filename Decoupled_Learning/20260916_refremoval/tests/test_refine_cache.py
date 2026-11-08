import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from refine_cache import CachedRefineDataset, load_cache_entries


def _write_rgb(path: Path, value: int) -> None:
    Image.new("RGB", (8, 8), (value, value, value)).save(path)


def test_cached_refine_dataset_keeps_all_modalities_aligned(tmp_path: Path):
    root = tmp_path / "prepared_refine"
    image_dir = root / "images" / "synthetic"
    prior_dir = root / "images" / "syntheticprior"
    coarse_dir = root / "coarse"
    image_dir.mkdir(parents=True)
    prior_dir.mkdir(parents=True)
    coarse_dir.mkdir(parents=True)

    sample_id = "synthetic_000000"
    input_path = image_dir / f"{sample_id}_blended.png"
    target_path = image_dir / f"{sample_id}_transmission.png"
    coarse_path = coarse_dir / f"{sample_id}_blended_diff.png"
    prior_path = prior_dir / f"{sample_id}_blended.npy"
    _write_rgb(input_path, 20)
    _write_rgb(target_path, 40)
    _write_rgb(coarse_path, 60)
    np.save(prior_path, np.full((8, 8), 0.5, dtype=np.float32))

    manifest_path = root / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({"id": sample_id, "input": str(input_path), "target": str(target_path)})
        + "\n",
        encoding="utf-8",
    )

    entries = load_cache_entries(str(root))
    dataset = CachedRefineDataset(
        entries,
        resolution=8,
        resize_scale=1.0,
        disable_augment=True,
    )
    sample = dataset[0]

    assert set(sample) == {"cond", "prelim", "gt", "prior"}
    assert sample["cond"].shape == (3, 8, 8)
    assert sample["prelim"].shape == (3, 8, 8)
    assert sample["gt"].shape == (3, 8, 8)
    assert sample["prior"].shape == (1, 8, 8)
    assert torch.isclose(sample["prior"].mean(), torch.tensor(0.5), atol=1 / 255)


def test_cache_loader_reports_missing_coarse_output(tmp_path: Path):
    root = tmp_path / "prepared_refine"
    image_dir = root / "images" / "synthetic"
    prior_dir = root / "images" / "syntheticprior"
    coarse_dir = root / "coarse"
    image_dir.mkdir(parents=True)
    prior_dir.mkdir(parents=True)
    coarse_dir.mkdir(parents=True)

    input_path = image_dir / "synthetic_000000_blended.png"
    target_path = image_dir / "synthetic_000000_transmission.png"
    _write_rgb(input_path, 20)
    _write_rgb(target_path, 40)
    np.save(prior_dir / "synthetic_000000_blended.npy", np.zeros((8, 8), dtype=np.float32))
    (root / "manifest.jsonl").write_text(
        json.dumps({"id": "synthetic_000000", "input": str(input_path), "target": str(target_path)})
        + "\n",
        encoding="utf-8",
    )

    try:
        load_cache_entries(str(root))
    except FileNotFoundError as error:
        assert "coarse" in str(error)
    else:
        raise AssertionError("Expected missing coarse output to raise FileNotFoundError")
