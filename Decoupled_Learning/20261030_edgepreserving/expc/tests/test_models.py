import sys
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import cv2

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from models import GUIDED_VARIANTS, VARIANTS, build_model
from train_expc import (RandomGuidedPatchPairs, RandomPatchPairs, append_train_loss,
                        ensure_float32_cache, ensure_lr_cache, protocol, seed_all)


def test_ddp_protocol_records_actual_global_batch(tmp_path):
    args = SimpleNamespace(
        parallelism="ddp", variant="viv_gf_e1", data_root=tmp_path,
        crop_size=128, batch_size_per_gpu=16, grad_accumulation=1,
        precision="bf16", lr=3e-4, weight_decay=1e-4, total_iter=60000,
        seed=42, loss_mode="base", init_lr_checkpoint=None,
        init_model_checkpoint=None, freeze_lr=False,
    )
    record = protocol(args, ["train"], ["val"])
    assert record["world_size"] == 4
    assert record["effective_batch_size"] == 64


def test_seed_all_seeds_only_current_cuda_device(monkeypatch):
    seeded_devices = []

    def unexpected_seed_all(_seed):
        raise AssertionError("CUDA seed_all must not touch every visible GPU")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", unexpected_seed_all)
    monkeypatch.setattr(torch.cuda, "manual_seed", seeded_devices.append)
    seed_all(42, 1)
    first = torch.rand(1)
    seed_all(42, 1)
    torch.testing.assert_close(torch.rand(1), first)
    assert seeded_devices == [43, 43]


def test_seed_all_preserves_data_parallel_cuda_seeding(monkeypatch):
    seeded_devices = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", seeded_devices.append)
    seed_all(42, 0, "dp")
    assert seeded_devices == [42]


def test_training_loss_csv_can_be_followed_across_updates(tmp_path):
    path = tmp_path / "train_loss.csv"
    append_train_loss(path, 1, 0.25, 0.20, 1.0, 3e-4)
    append_train_loss(path, 100, 0.15, 0.12, 0.6, 2.9e-4)
    rows = path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "iteration,loss,pixel,gradient,lr"
    assert len(rows) == 3
    assert rows[-1].startswith("100,0.15,0.12,0.6,")


def test_all_expc_models_preserve_image_shape_and_backpropagate():
    torch.set_num_threads(2)
    image = torch.rand(2, 1, 128, 128)
    for variant in VARIANTS:
        model = build_model(variant)
        prediction = model(image, torch.rand(2, 1, 64, 64)) if variant in GUIDED_VARIANTS else model(image)
        assert prediction.shape == image.shape
        assert torch.isfinite(prediction).all()
        (prediction - image).abs().mean().backward()
        assert any(parameter.grad is not None for parameter in model.parameters())


def test_guided_pyramid_changes_routing_without_changing_weights():
    flat = build_model("deep_guided_lr4").eval()
    pyramid = build_model("guided_pyramid_lr4").eval()
    assert flat.state_dict().keys() == pyramid.state_dict().keys()
    assert sum(p.numel() for p in flat.parameters()) == sum(p.numel() for p in pyramid.parameters())
    pyramid.load_state_dict(flat.state_dict(), strict=True)
    lr = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        assert not torch.equal(flat.predict_lr(lr), pyramid.predict_lr(lr))


def test_structured_models_expose_interpretable_maps():
    image = torch.rand(1, 1, 128, 128)
    expected = {
        "similarity_pyramid": {"similarity_gate", "edge_guide"},
        "wls_naf": {"similarity_gate_half", "similarity_gate_full", "edge_guide", "edge_gate"},
        "lowres_edge": {"base", "high_frequency", "edge_gate"},
        "unrolled_diffusion": {
            "data_term", "horizontal_conductance", "vertical_conductance",
            "lambda", "iterations",
        },
    }
    for variant, keys in expected.items():
        prediction, diagnostics = build_model(variant)(image, return_aux=True)
        assert prediction.shape == image.shape
        assert keys <= diagnostics.keys()


def test_training_dataset_crops_from_cached_float32_array(tmp_path):
    cache_path = tmp_path / "pairs.npy"
    cache = torch.arange(6 * 7, dtype=torch.float32).reshape(1, 6, 7).numpy()
    mapped = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype="float32", shape=(1, 2, 6, 7)
    )
    mapped[0, 0] = cache
    mapped[0, 1] = cache * 2
    mapped.flush()
    del mapped

    dataset = RandomPatchPairs(cache_path, [0], crop_size=4)
    image, target = dataset[0]
    assert isinstance(dataset._cache, np.memmap)
    assert image.shape == target.shape == (1, 4, 4)
    assert torch.equal(target, image * 2)


def test_float32_cache_decodes_each_pair_into_reusable_map(tmp_path):
    class Pair:
        ids = ["sample-0", "sample-1"]

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, index):
            image = torch.full((1, 5, 6), float(index), dtype=torch.float32)
            return image, image + 0.25

    cache_path = ensure_float32_cache(Pair(), tmp_path / "train.npy", 0, "dp")
    mapped = np.load(cache_path, mmap_mode="r")
    assert mapped.shape == (2, 2, 5, 6)
    assert mapped.dtype == np.float32
    np.testing.assert_array_equal(mapped[1, 0], np.ones((5, 6), dtype=np.float32))
    np.testing.assert_array_equal(mapped[1, 1], np.full((5, 6), 1.25, dtype=np.float32))


def test_guided_patch_dataset_keeps_lr_hr_crop_alignment(tmp_path):
    hr_path = tmp_path / "hr.npy"
    lr_path = tmp_path / "lr.npy"
    hr = np.arange(1 * 2 * 8 * 8, dtype=np.float32).reshape(1, 2, 8, 8)
    lr = np.arange(1 * 1 * 4 * 4, dtype=np.float32).reshape(1, 1, 4, 4)
    np.save(hr_path, hr)
    np.save(lr_path, lr)
    dataset = RandomGuidedPatchPairs(hr_path, lr_path, [0], crop_size=4)
    low, image, target = dataset[0]
    assert low.shape == (1, 2, 2)
    assert image.shape == target.shape == (1, 4, 4)
    top, left = int(image[0, 0, 0].item() // 8), int(image[0, 0, 0].item() % 8)
    assert top % 2 == left % 2 == 0
    assert torch.equal(target, image + 64)
    assert low[0, 0, 0] == lr[0, 0, top // 2, left // 2]


def test_lr_cache_uses_paired_downsample_tiff_and_metadata(tmp_path):
    root = tmp_path / "data"
    lr_dir = root / "traininput-l0p500-a1p400_downsample"
    lr_dir.mkdir(parents=True)
    source = np.arange(16, dtype=np.float32).reshape(4, 4)
    cv2.imwrite(str(lr_dir / "sample-log.tiff"), source)

    class Pair:
        ids = ["sample"]
        pairs = [(Path("traininput/sample-log.tiff"), Path("label/sample-WLS.tiff"))]

        def __len__(self):
            return 1

    cache_path = ensure_lr_cache(Pair(), root, tmp_path / "lr.npy", 0, "dp")
    mapped = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    assert mapped.shape == (1, 1, 4, 4)
    np.testing.assert_array_equal(mapped[0, 0], source)
    metadata = json.loads(cache_path.with_suffix(".npy.json").read_text())
    assert metadata["scale"] == 2
