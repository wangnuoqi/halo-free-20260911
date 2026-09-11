from evaluate_20260819 import build_test_paths, mean_metrics


def test_build_test_paths_uses_20260818_test_split():
    paths = build_test_paths(
        "/mnt/d/nuoqi/S2R-HDR/data/20260818",
        "/mnt/d/nuoqi/halo-free/Decoupled_Learning/20260815_edgepreserving",
    )

    assert paths["lr"].name == "testinput-l0p500-a1p400-200_downsample"
    assert paths["origin"].name == "testinput-l0p500-a1p400-200"
    assert paths["label"].name == "testlabel-l0p500-a1p400-200"


def test_mean_metrics_averages_per_image_mse_and_psnr():
    metrics = mean_metrics([
        {"mse": 0.25, "psnr": 6.0},
        {"mse": 0.5, "psnr": 4.0},
    ])

    assert metrics["mse"] == 0.375
    assert metrics["psnr"] == 5.0
