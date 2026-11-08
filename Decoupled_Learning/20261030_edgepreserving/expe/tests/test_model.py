import pytest
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from expe.model import (EXPE_VARIANTS, HRNeighborFusion, build_expe_model,
                        edge_features, otsu_edge_map)
from expe.losses import experiment_loss
from expe.checkpoint import initialize_lr, initialize_model
from source_snapshot.baseline.vivnet_wls.model import build_model as build_vivnet


def test_e1_ddp_completes_two_steps_with_upstream_unused_weight(tmp_path):
    torch.set_num_threads(2)
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'ddp_init'}",
                            rank=0, world_size=1)
    try:
        model = DDP(build_expe_model("viv_gf_e1"), find_unused_parameters=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        guide = torch.rand(1, 1, 32, 36)
        lr = torch.rand(1, 1, 16, 18)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            prediction, auxiliary = model(guide, lr, return_aux=True)
            loss, _, _ = experiment_loss(prediction.float(), guide, auxiliary, "base")
            loss.backward()
            optimizer.step()
        assert model.module.lr.reduce_chan_level2.weight.grad is None
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("variant", EXPE_VARIANTS)
def test_variants_keep_two_to_one_geometry_and_backpropagate(variant):
    torch.set_num_threads(2)
    model = build_expe_model(variant)
    lr = torch.rand(1, 1, 16, 18)
    guide = torch.rand(1, 1, 32, 36)
    prediction, aux = model(guide, lr, return_aux=True)
    assert prediction.shape == guide.shape
    assert torch.isfinite(prediction).all()
    assert aux["y_lr"].shape == lr.shape
    assert len(aux["stages"]) == model.stages + 1
    prediction.mean().backward()
    assert any(p.grad is not None for p in model.gf.parameters())


def test_multi_box_weights_are_normalized_and_candidates_are_distinct():
    model = build_expe_model("viv_gf_b")
    guide = torch.rand(1, 1, 32, 36)
    _, aux = model(guide, torch.rand(1, 1, 16, 18), return_aux=True)
    assert aux["box_weights"].shape == (1, 2, 16, 18)
    torch.testing.assert_close(aux["box_weights"].sum(1), torch.ones(1, 16, 18))
    assert aux["candidate_a"].shape == (1, 2, 16, 18)
    assert not torch.allclose(aux["candidate_a"][:, 0], aux["candidate_a"][:, 1])


def test_lr_predictor_keeps_vivnet_unet_without_refinement_blocks():
    predictor = build_expe_model("viv_gf_e1").lr
    reference = build_vivnet(torch.device("cpu"))
    assert len(predictor.refinement) == 0
    assert predictor.patch_embed.proj.out_channels == 32
    assert {name: value.shape for name, value in predictor.state_dict().items()} == {
        name: value.shape for name, value in reference.state_dict().items()
        if not name.startswith("refinement.")
    }


def test_e0_reproduces_legacy_lrnet_layout_and_current_training_interface():
    model = build_expe_model("exp7c_e0")
    assert model.lr.stem[0].in_channels == 1
    assert model.lr.stem[0].out_channels == 24
    assert [block[0].dilation for block in model.lr.dilated_blocks] == [(2, 2), (4, 4), (8, 8), (16, 16)]
    guide = torch.rand(2, 1, 32, 36)
    lr = torch.rand(2, 1, 16, 18)
    prediction, aux = model(guide, lr, return_aux=True)
    assert prediction.shape == guide.shape
    assert torch.isfinite(prediction).all()
    assert aux["y_lr"].shape == lr.shape
    assert len(aux["stages"]) == 1
    prediction.mean().backward()
    assert any(parameter.grad is not None for parameter in model.lr.parameters())
    assert any(parameter.grad is not None for parameter in model.gf.parameters())


def test_otsu_map_is_finite_and_binary_on_constant_input():
    edges = otsu_edge_map(torch.ones(2, 1, 12, 14))
    assert edges.shape == (2, 1, 12, 14)
    assert torch.count_nonzero(edges) == 0


def test_edge_feature_ablation_omits_unused_cues():
    guide = torch.rand(1, 1, 32, 36)
    y_lr = torch.rand(1, 1, 16, 18)
    gradient, _ = edge_features(guide, y_lr, "gradient")
    input_only, _ = edge_features(guide, y_lr, "input")
    assert torch.count_nonzero(gradient[:, 1::2]) == 0
    assert torch.count_nonzero(input_only[:, 2:]) == 0


def test_hr_fusion_initializes_close_to_bilinear_coefficients():
    fusion = HRNeighborFusion()
    a, b, x, y = [torch.rand(1, 1, 8, 9) for _ in range(4)]
    guide = torch.rand(1, 1, 16, 18)
    a_hr, b_hr, weights, blend = fusion(a, b, x, y, guide, guide.new_zeros(1, 4, 16, 18))
    for actual, low in ((a_hr, a), (b_hr, b)):
        baseline = F.interpolate(low, size=guide.shape[-2:], mode="bilinear", align_corners=True)
        assert (actual - baseline).abs().max() < 0.03
    torch.testing.assert_close(weights.sum(1), torch.ones(1, 16, 18))
    assert blend.max() < 0.03


def test_reconstruction_can_hold_lr_prediction_fixed_for_oracle_diagnostic():
    model = build_expe_model("viv_gf_a1_stats").eval()
    guide = torch.rand(1, 1, 32, 36)
    lr = torch.rand(1, 1, 16, 18)
    low = torch.zeros_like(lr)
    high = torch.ones_like(lr)
    with torch.no_grad():
        output_low, auxiliary = model.reconstruct(guide, lr, low, return_aux=True)
        output_high = model.reconstruct(guide, lr, high)
    torch.testing.assert_close(auxiliary["y_lr"], low)
    assert not torch.allclose(output_low, output_high)


def test_c2_updates_both_coefficients_and_supports_staged_loss():
    model = build_expe_model("viv_gf_c2")
    guide = torch.rand(1, 1, 32, 36)
    lr = torch.rand(1, 1, 16, 18)
    target = torch.rand_like(guide)
    prediction, aux = model(guide, lr, return_aux=True)
    assert len(aux["stages"]) == 3
    assert len(aux["delta_a"]) == len(aux["delta_b"]) == 2
    loss, pixel, gradient = experiment_loss(prediction, target, aux, "staged_regularized")
    assert all(torch.isfinite(value) for value in (loss, pixel, gradient))
    loss.backward()
    assert any(p.grad is not None for p in model.updates[0].parameters())


def test_unsupported_mode_and_wrong_geometry_are_rejected():
    model = build_expe_model("viv_gf_a1_stats")
    with pytest.raises(ValueError, match="2x"):
        model(torch.rand(1, 1, 31, 32), torch.rand(1, 1, 16, 16))
    with pytest.raises(ValueError, match="loss mode"):
        experiment_loss(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8), {}, "unknown")


def test_initialize_lr_uses_only_e1_checkpoint_and_freezes_predictor(tmp_path):
    source = build_expe_model("viv_gf_e1")
    destination = build_expe_model("viv_gf_a1_stats")
    path = tmp_path / "best.pth"
    torch.save({"model": source.state_dict(),
                "protocol": {"variant": "viv_gf_e1"}}, path)
    initialize_lr(destination, path, freeze=True)
    for name, tensor in source.lr.state_dict().items():
        torch.testing.assert_close(destination.lr.state_dict()[name], tensor)
    assert not any(parameter.requires_grad for parameter in destination.lr.parameters())
    assert destination.lr.training is False
    destination.train()
    assert destination.lr.training is False


def test_initialize_model_loads_complete_variant_for_joint_tuning(tmp_path):
    source = build_expe_model("viv_gf_c2")
    destination = build_expe_model("viv_gf_c2")
    path = tmp_path / "best.pth"
    torch.save({"model": source.state_dict(),
                "protocol": {"variant": "viv_gf_c2"}}, path)
    initialize_model(destination, path, "viv_gf_c2")
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(destination.state_dict()[name], tensor)
    assert all(parameter.requires_grad for parameter in destination.lr.parameters())
