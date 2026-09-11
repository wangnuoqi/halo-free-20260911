import torch

from exp1.nyuv2_full_dataset import NYUv2FullImageDataset
from exp3.train_exp3_joint import DCTPatchDataset, DKNArrayDataset, build_official_scheduler
from thirdparty_refine_models import Exp6DRefine3Stage


def test_exp6d_joint_rf_forward_shape():
    model = Exp6DRefine3Stage()
    out = model(torch.rand(1, 3, 32, 40), torch.rand(1, 1, 32, 40))
    assert out["final"].shape == (1, 1, 32, 40)


def test_exp3_dkn_array_dataset_has_official_split():
    ds = DKNArrayDataset("exp1/data/dkn_nyuv2_1000_449", train=True)
    depth, rgb, target, mn, mx = ds[0]
    assert len(ds) == 1000
    assert depth.shape == (1, 60, 80)
    assert rgb.shape == (3, 480, 640)
    assert target.shape == (1, 480, 640)
    assert float(mx) >= float(mn)


def test_exp3_dct_patch_dataset_has_256_patches():
    ds = DCTPatchDataset("/mnt/d/nuoqi/halo-free/data_NYUv2/NYUv2-resplit/NYUv2")
    low, rgb, target, _, _ = ds[0]
    assert len(ds) > 0
    assert low.shape == (1, 32, 32)
    assert rgb.shape == (3, 256, 256)
    assert target.shape == (1, 256, 256)


def test_conv_full_image_dataset_has_optional_name_field():
    ds = NYUv2FullImageDataset(
        "/mnt/d/nuoqi/halo-free/data_NYUv2/NYUv2-resplit/NYUv2",
        "train", 8, indices=range(1), depth_unit_scale=1000.0,
    )
    assert len(ds[0]) == 6


def test_official_scheduler_protocols():
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=1e-3)
    scheduler = build_official_scheduler("dkn", optimizer)
    assert scheduler.step_size == 10000
    assert scheduler.gamma == 0.2
    scheduler = build_official_scheduler("dctnet", optimizer)
    assert scheduler.step_size == 100
    assert scheduler.gamma == 0.5
