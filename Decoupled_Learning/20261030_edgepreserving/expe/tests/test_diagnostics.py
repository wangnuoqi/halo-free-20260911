import torch

from expe.diagnostics import region_metrics


def test_region_metrics_separates_target_edges_from_flat_area():
    target = torch.zeros(1, 1, 32, 32)
    target[..., 16:] = 1
    prediction = target.clone()
    prediction[..., 15:17] = 0.5
    metrics = region_metrics(prediction, target)
    assert metrics["edge_pixels"] > 0
    assert metrics["flat_pixels"] > 0
    assert metrics["edge_mae"] > metrics["flat_mae"]
    assert metrics["mse"] > 0


def test_region_metrics_handles_flat_target():
    target = torch.zeros(1, 1, 16, 16)
    metrics = region_metrics(target, target)
    assert metrics["edge_pixels"] == 0
    assert metrics["flat_pixels"] == 256
    assert metrics["mse"] == metrics["flat_mae"] == 0
