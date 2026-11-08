import copy
import unittest

import torch
from torch.nn import functional as F

from exp_b.src.model_exp_b import ConvGuidedFilter
from expe.learned_affine import LearnedKernelAffine
from expe.model import EXPE_VARIANTS, MultiBoxAffine, build_expe_model
from expe.losses import route_supervision_loss


class LearnedAffineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_small_branch_can_reproduce_e1(self):
        original = ConvGuidedFilter().eval()
        model = LearnedKernelAffine(direct=False).eval()
        model.branches[0].load_state_dict(copy.deepcopy(original.state_dict()))
        with torch.no_grad():
            model.gate[-1].bias.copy_(torch.tensor([80., -80.]))
        x, y, guide = torch.rand(2, 1, 12, 16) - .5, torch.rand(2, 1, 12, 16), torch.rand(2, 1, 24, 32) - .5
        a, b, _ = model(x, y)
        actual = (F.interpolate(a, guide.shape[-2:], mode='bilinear', align_corners=True) * guide +
                  F.interpolate(b, guide.shape[-2:], mode='bilinear', align_corners=True)).clamp(0, 1)
        torch.testing.assert_close(actual, original(x, y, guide), rtol=0, atol=0)

    def test_both_estimators_and_gate_receive_gradients(self):
        for direct in (False, True):
            model = LearnedKernelAffine(direct=direct)
            a, b, aux = model(torch.rand(2, 1, 12, 16) - .5, torch.rand(2, 1, 12, 16))
            (a.square().mean() + b.square().mean()).backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, (direct, name))
                self.assertTrue(torch.isfinite(parameter.grad).all(), (direct, name))
            torch.testing.assert_close(aux['box_weights'].sum(1), torch.ones(2, 12, 16))

    def test_registered_models_reconstruct_and_train_lrnet(self):
        for variant in ('viv_gf_a1_stats', 'viv_gf_a2_stats_edge',
                        'viv_gf_a1_direct', 'viv_gf_a2_direct_edge',
                        'viv_gf_a1_stats_freeb', 'viv_gf_a1_stats_edgebranch',
                        'viv_gf_a1_stats_routesup'):
            model = build_expe_model(variant)
            prediction, aux = model(torch.rand(2, 1, 32, 32), torch.rand(2, 1, 16, 16), return_aux=True)
            self.assertEqual(prediction.shape, (2, 1, 32, 32))
            prediction.square().mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.lr.parameters()))
            self.assertEqual(aux['candidate_a'].shape, (2, 2, 16, 16))

    def test_official_a_names_and_archived_names_are_distinct(self):
        official = {'viv_gf_a1_stats', 'viv_gf_a2_stats_edge',
                    'viv_gf_a1_direct', 'viv_gf_a2_direct_edge',
                    'viv_gf_a1_stats_freeb', 'viv_gf_a1_stats_edgebranch',
                    'viv_gf_a1_stats_routesup',
                    'viv_gf_a1_fixed_boxes', 'viv_gf_a1_box3', 'viv_gf_a1_box9'}
        self.assertTrue(official.issubset(EXPE_VARIANTS))
        self.assertFalse({'viv_gf_a1', 'viv_gf_a2', 'viv_gf_a2_grad',
                          'viv_gf_a2_otsu', 'viv_gf_a2_input'} & EXPE_VARIANTS.keys())
        self.assertIsInstance(build_expe_model('viv_gf_a1').gf, MultiBoxAffine)

    def test_freeb_statistics_branch_outputs_independent_affine_pair(self):
        model = LearnedKernelAffine(free_b=True).eval()
        self.assertEqual(model.branches[0].a[-1].out_channels, 2)
        x = torch.rand(2, 1, 12, 16) - .5
        y = torch.rand_like(x)
        _, _, aux = model(x, y)
        self.assertEqual(aux['candidate_a'].shape, (2, 2, 12, 16))
        self.assertEqual(aux['candidate_b'].shape, (2, 2, 12, 16))

    def test_edgebranch_features_enter_statistics_coefficient_head(self):
        model = LearnedKernelAffine(branch_edges=True).eval()
        self.assertEqual(model.branches[0].a[0].in_channels, 6)
        x = torch.rand(2, 1, 12, 16) - .5
        y = torch.rand_like(x)
        edges = torch.rand(2, 4, 12, 16)
        a, b, aux = model(x, y, edges)
        self.assertEqual(a.shape, x.shape)
        self.assertEqual(b.shape, x.shape)
        self.assertEqual(aux['candidate_a'].shape, (2, 2, 12, 16))

    def test_route_supervision_uses_candidate_gt_errors_and_backpropagates_gate(self):
        model = LearnedKernelAffine(route_supervision=True)
        guide = torch.rand(2, 1, 24, 32)
        x = torch.rand(2, 1, 12, 16) - .5
        y = torch.rand_like(x)
        _, _, aux = model(x, y)
        loss = route_supervision_loss(aux, guide, torch.rand_like(guide))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.gate[-1].weight.grad)

    def test_edge_variants_feed_four_edge_maps_to_gate(self):
        for direct in (False, True):
            model = LearnedKernelAffine(direct=direct, use_edges=True).eval()
            x = torch.rand(2, 1, 12, 16) - .5
            y = torch.rand_like(x)
            edges = torch.ones(2, 4, 12, 16)
            with torch.no_grad():
                model.gate[0].weight.zero_()
                model.gate[0].bias.zero_()
                model.gate[0].weight[0, -1].fill_(1)
                model.gate[-1].weight.zero_()
                model.gate[-1].weight[0, 0].fill_(1)
            _, _, without_edges = model(x, y, torch.zeros_like(edges))
            _, _, with_edges = model(x, y, edges)
            self.assertFalse(torch.equal(without_edges['box_weights'], with_edges['box_weights']))

    def test_zero_edge_maps_preserve_base_candidate_path(self):
        for direct in (False, True):
            base = LearnedKernelAffine(direct=direct).eval()
            edge = LearnedKernelAffine(direct=direct, use_edges=True).eval()
            edge.branches.load_state_dict(copy.deepcopy(base.branches.state_dict()))
            with torch.no_grad():
                edge.gate[0].weight.zero_()
                edge.gate[0].weight[:, :base.gate[0].in_channels].copy_(base.gate[0].weight)
                edge.gate[0].bias.copy_(base.gate[0].bias)
                edge.gate[-1].load_state_dict(copy.deepcopy(base.gate[-1].state_dict()))
            x = torch.rand(2, 1, 12, 16) - .5
            y = torch.rand_like(x)
            with torch.no_grad():
                expected = base(x, y)
                actual = edge(x, y, torch.zeros(2, 4, 12, 16))
            for expected_part, actual_part in zip(expected[:2], actual[:2]):
                torch.testing.assert_close(actual_part, expected_part, rtol=0, atol=0)
            torch.testing.assert_close(actual[2]['box_weights'], expected[2]['box_weights'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
