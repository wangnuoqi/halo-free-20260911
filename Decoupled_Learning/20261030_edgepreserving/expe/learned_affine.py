"""Learned spatial kernels with a shared gate for each affine pair."""

import torch
from torch import nn

from exp_b.src.model_exp_b import AdaptiveNorm, ConvGuidedFilter


class LearnedKernelAffine(nn.Module):
    def __init__(self, direct=False, use_edges=False, free_b=False,
                 branch_edges=False, route_supervision=False):
        super().__init__()
        self.direct = direct
        self.use_edges = use_edges
        self.free_b = free_b
        self.branch_edges = branch_edges
        self.route_supervision = route_supervision
        if direct and (free_b or branch_edges or route_supervision):
            raise ValueError("free_b, branch_edges and route_supervision are only defined for statistics branches")
        self.branches = nn.ModuleList()
        for size in (3, 9):
            if direct:
                branch = nn.Sequential(
                    nn.Conv2d(2, 32, size, padding=size // 2, bias=False),
                    AdaptiveNorm(32), nn.ReLU(inplace=False),
                    nn.Conv2d(32, 32, 1, bias=False),
                    AdaptiveNorm(32), nn.ReLU(inplace=False),
                    nn.Conv2d(32, 2, 1, bias=False),
                )
            else:
                branch = ConvGuidedFilter()
                if size != 3:
                    branch.box = nn.Conv2d(1, 1, size, padding=size // 2, bias=False)
                    nn.init.ones_(branch.box.weight)
                if self.branch_edges:
                    old_input = branch.a[0]
                    expanded_input = nn.Conv2d(6, old_input.out_channels, 1, bias=False)
                    with torch.no_grad():
                        expanded_input.weight[:, :2].copy_(old_input.weight)
                        expanded_input.weight[:, 2:].zero_()
                    branch.a[0] = expanded_input
                if self.free_b:
                    old_output = branch.a[-1]
                    expanded_output = nn.Conv2d(old_output.in_channels, 2, 1, bias=False)
                    with torch.no_grad():
                        expanded_output.weight[0].copy_(old_output.weight[0])
                        expanded_output.weight[1].zero_()
                    branch.a[-1] = expanded_output
            self.branches.append(branch)
        gate_inputs = (6 if direct else 8) + (4 if use_edges else 0)
        if route_supervision:
            # Per-branch affine parameters and their LR-grid candidate outputs.
            gate_inputs += 6
        self.gate = nn.Sequential(
            nn.Conv2d(gate_inputs, 16, 1),
            nn.GELU(), nn.Conv2d(16, 2, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, x_lr, y_lr, edges_lr=None):
        if self.branch_edges or self.use_edges:
            if edges_lr is None or edges_lr.shape != (x_lr.shape[0], 4, *x_lr.shape[-2:]):
                raise ValueError("Edge-guided affine estimation requires four aligned LR edge maps")
        slopes, intercepts, statistics = [], [], []
        for branch in self.branches:
            if self.direct:
                a, b = branch(torch.cat((x_lr, y_lr), dim=1)).chunk(2, dim=1)
            else:
                # Exactly the E1 operations, including its learned normalizer.
                count = branch.box(torch.ones_like(x_lr))
                mean_x = branch.box(x_lr) / count
                mean_y = branch.box(y_lr) / count
                covariance = branch.box(x_lr * y_lr) / count - mean_x * mean_y
                variance = branch.box(x_lr * x_lr) / count - mean_x * mean_x
                branch_input = torch.cat((covariance, variance), dim=1)
                if self.branch_edges:
                    branch_input = torch.cat((branch_input, edges_lr), dim=1)
                branch_output = branch.a(branch_input)
                if self.free_b:
                    a, b = branch_output.chunk(2, dim=1)
                else:
                    a = branch_output
                    b = mean_y - a * mean_x
                statistics.extend((mean_x, mean_y, covariance, variance))
            slopes.append(a)
            intercepts.append(b)
        candidate_a, candidate_b = torch.cat(slopes, 1), torch.cat(intercepts, 1)
        features = (torch.cat((x_lr, y_lr, candidate_a, candidate_b), 1)
                    if self.direct else torch.cat(statistics, 1))
        if self.route_supervision:
            candidate_lr = candidate_a * x_lr + candidate_b
            features = torch.cat((features, candidate_a, candidate_b, candidate_lr), dim=1)
        if self.use_edges:
            features = torch.cat((features, edges_lr), dim=1)
        weights = self.gate(features).softmax(dim=1)
        # The SAME weights choose the pair, preserving each candidate's affine relation.
        a = (weights * candidate_a).sum(1, keepdim=True)
        b = (weights * candidate_b).sum(1, keepdim=True)
        return a, b, {"box_weights": weights, "candidate_a": candidate_a,
                      "candidate_b": candidate_b}
