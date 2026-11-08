"""VIV-LR predictor and staged affine-reconstruction ablations for Exp E."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from exp_b.src.model_exp_b import ConvGuidedFilter
from expe.archive.legacy_a.variants import LEGACY_A_VARIANTS
from expe.learned_affine import LearnedKernelAffine
from source_snapshot.baseline.vivnet_wls.VIVNet_arch import VIVNet
from source_snapshot.baseline.vivnet_wls.model import run_padded


EXPE_VARIANTS = {
    "exp7c_e0": {},
    "viv_gf_e1": dict(multi_box=False),
    "viv_gf_a1_box3": dict(boxes=(3,)),
    "viv_gf_a1_box9": dict(boxes=(9,)),
    "viv_gf_a1_fixed_boxes": dict(),
    "viv_gf_a1_stats": dict(affine_estimator="learned_stats"),
    "viv_gf_a2_stats_edge": dict(affine_estimator="learned_stats", use_edges=True),
    "viv_gf_a1_direct": dict(affine_estimator="direct"),
    "viv_gf_a2_direct_edge": dict(affine_estimator="direct", use_edges=True),
    "viv_gf_a1_stats_freeb": dict(affine_estimator="learned_stats", free_b=True),
    "viv_gf_a1_stats_edgebranch": dict(affine_estimator="learned_stats", branch_edges=True),
    "viv_gf_a1_stats_routesup": dict(affine_estimator="learned_stats",
                                      route_supervision=True),
    "viv_gf_b": dict(use_edges=True, hr_fusion=True),
    "viv_gf_c1": dict(use_edges=True, hr_fusion=True, stages=1),
    "viv_gf_c2_aonly": dict(use_edges=True, hr_fusion=True, stages=2, update_b=False),
    "viv_gf_c2": dict(use_edges=True, hr_fusion=True, stages=2),
    "viv_gf_c3": dict(use_edges=True, hr_fusion=True, stages=3),
}


class LegacyAdaptiveNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.ones(1))
        self.w1 = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(channels, momentum=0.999, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w0 * x + self.w1 * self.bn(x)


def _legacy_identity_init(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        with torch.no_grad():
            module.weight.zero_()
            out_channels, in_channels, kh, kw = module.weight.shape
            if out_channels < in_channels:
                nn.init.xavier_uniform_(module.weight)
            else:
                module.weight[:, :, kh // 2, kw // 2].diagonal().fill_(1.0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class LegacyExp7CLRNet(nn.Module):
    """The original 24-channel, four-dilation LRNet used by Exp7C."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1, bias=False),
            LegacyAdaptiveNorm(24),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dilated_blocks = nn.Sequential(*(
            nn.Sequential(
                nn.Conv2d(24, 24, 3, padding=2 ** level,
                          dilation=2 ** level, bias=False),
                LegacyAdaptiveNorm(24),
                nn.LeakyReLU(0.2, inplace=True),
            ) for level in range(1, 5)
        ))
        self.tail = nn.Sequential(
            nn.Conv2d(24, 24, 3, padding=1, bias=False),
            LegacyAdaptiveNorm(24),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(24, 1, 1),
        )
        self.apply(_legacy_identity_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.dilated_blocks(self.stem(x)))


class Exp7CE0(nn.Module):
    """Legacy Exp7C coarse model trained with the current Exp E protocol."""

    stages = 0

    def __init__(self):
        super().__init__()
        self.lr = LegacyExp7CLRNet()
        self.gf = ConvGuidedFilter()

    def forward(self, guide: torch.Tensor, lr: torch.Tensor, return_aux: bool = False):
        if guide.shape[-2:] != (2 * lr.shape[-2], 2 * lr.shape[-1]):
            raise ValueError("Exp7C E0 requires a 2x HR guide and LR input")
        lr_centered, guide_centered = lr - 0.5, guide - 0.5
        y_lr = self.lr(lr_centered)
        result = self.gf(lr_centered, y_lr, guide_centered)
        if return_aux:
            return result, {"y_lr": y_lr, "stages": [result]}
        return result


def gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    image = image.float()
    kernel_x = image.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-2, -1)
    padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
    dx = F.conv2d(padded, kernel_x)
    dy = F.conv2d(padded, kernel_y)
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


@torch.no_grad()
def otsu_edge_map(image: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """Otsu threshold of per-image normalized gradient, with no GT dependency."""
    magnitude = gradient_magnitude(image)
    peak = magnitude.amax(dim=(-2, -1), keepdim=True)
    normalized = magnitude / peak.clamp_min(1e-6)
    masks = []
    for sample, sample_peak in zip(normalized, peak):
        histogram = torch.histc(sample.float(), bins=bins, min=0.0, max=1.0)
        probability = histogram / histogram.sum().clamp_min(1)
        mass = probability.cumsum(0)
        mean = (probability * torch.arange(bins, device=image.device)).cumsum(0)
        total = mean[-1]
        score = (total * mass - mean).square() / (mass * (1 - mass)).clamp_min(1e-8)
        threshold = (score.argmax() + 0.5).float() / bins
        masks.append(((sample > threshold) & (sample_peak > 1e-6)).float())
    return torch.stack(masks, dim=0).to(image.dtype)


def edge_features(guide: torch.Tensor, y_lr: torch.Tensor,
                  mode: str = "both") -> tuple[torch.Tensor, torch.Tensor]:
    """Return HR and LR edge cues from inference-available inputs only."""
    if mode not in {"both", "gradient", "otsu", "input"}:
        raise ValueError(f"Unknown edge feature mode: {mode}")
    y_hr = F.interpolate(y_lr, size=guide.shape[-2:], mode="bilinear", align_corners=True)
    blank = torch.zeros_like(guide, dtype=torch.float32)
    use_gradient, use_otsu = mode in {"both", "gradient", "input"}, mode in {"both", "otsu", "input"}
    use_target = mode != "input"
    hr = torch.cat((gradient_magnitude(guide) if use_gradient else blank,
                    otsu_edge_map(guide) if use_otsu else blank,
                    gradient_magnitude(y_hr) if use_target and use_gradient else blank,
                    otsu_edge_map(y_hr) if use_target and use_otsu else blank), dim=1)
    return hr, F.avg_pool2d(hr, 2, stride=2)


class MultiBoxAffine(nn.Module):
    """Estimate affine candidates on one LR grid and allocate box scales per pixel."""

    def __init__(self, use_edges: bool, boxes: tuple[int, ...]):
        super().__init__()
        self.use_edges = use_edges
        self.boxes = boxes
        self.slope = nn.Sequential(nn.Conv2d(2, 32, 1), nn.GELU(),
                                   nn.Conv2d(32, 1, 1))
        self.allocate = nn.Sequential(
            nn.Conv2d(4 * len(boxes) + (4 if use_edges else 0), 16, 1), nn.GELU(),
            nn.Conv2d(16, len(boxes), 1),
        )
        nn.init.zeros_(self.allocate[-1].weight)
        nn.init.zeros_(self.allocate[-1].bias)

    def forward(self, x_lr: torch.Tensor, y_lr: torch.Tensor,
                edges_lr: torch.Tensor | None = None):
        slopes, intercepts, statistics = [], [], []
        for size in self.boxes:
            mean_x = F.avg_pool2d(x_lr, size, stride=1, padding=size // 2,
                                   count_include_pad=False)
            mean_y = F.avg_pool2d(y_lr, size, stride=1, padding=size // 2,
                                   count_include_pad=False)
            covariance = (F.avg_pool2d(x_lr * y_lr, size, stride=1,
                                      padding=size // 2, count_include_pad=False)
                          - mean_x * mean_y)
            variance = (F.avg_pool2d(x_lr.square(), size, stride=1,
                                    padding=size // 2, count_include_pad=False)
                        - mean_x.square()).clamp_min(0)
            slope = self.slope(torch.cat((covariance, variance), dim=1))
            slopes.append(slope)
            intercepts.append(mean_y - slope * mean_x)
            statistics.extend((mean_x, mean_y, covariance, variance))
        features = torch.cat(statistics + ([edges_lr] if self.use_edges else []), dim=1)
        weights = self.allocate(features).softmax(dim=1)
        candidate_a = torch.cat(slopes, dim=1)
        candidate_b = torch.cat(intercepts, dim=1)
        a = (weights * candidate_a).sum(dim=1, keepdim=True)
        b = (weights * candidate_b).sum(dim=1, keepdim=True)
        return a, b, {"box_weights": weights, "candidate_a": candidate_a,
                      "candidate_b": candidate_b}


def _neighbor(x: torch.Tensor, down: int, right: int) -> torch.Tensor:
    shifted = F.pad(x, (0, right, 0, down), mode="replicate")
    shifted = shifted[..., down:down + x.shape[-2], right:right + x.shape[-1]]
    return F.interpolate(shifted, scale_factor=2, mode="nearest")


class HRNeighborFusion(nn.Module):
    """Select four adjacent LR affine models using HR and predicted-target cues."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Sequential(nn.Conv2d(30, 16, 1), nn.GELU(),
                                    nn.Conv2d(16, 5, 1))
        nn.init.zeros_(self.logits[-1].weight)
        nn.init.zeros_(self.logits[-1].bias)
        self.logits[-1].bias.data[4] = -4.0

    def forward(self, a: torch.Tensor, b: torch.Tensor, x_lr: torch.Tensor,
                y_lr: torch.Tensor, guide: torch.Tensor, edges_hr: torch.Tensor):
        up_y = F.interpolate(y_lr, size=guide.shape[-2:], mode="bilinear", align_corners=True)
        groups, slopes, intercepts = [], [], []
        for down, right in ((0, 0), (0, 1), (1, 0), (1, 1)):
            local_a, local_b = _neighbor(a, down, right), _neighbor(b, down, right)
            local_x, local_y = _neighbor(x_lr, down, right), _neighbor(y_lr, down, right)
            groups.extend((local_a, local_b, local_x, local_y,
                           (guide - local_x).abs(), (up_y - local_y).abs()))
            slopes.append(local_a)
            intercepts.append(local_b)
        features = torch.cat((guide, up_y, edges_hr, *groups), dim=1)
        raw = self.logits(features)
        weights, blend = raw[:, :4].softmax(dim=1), raw[:, 4:5].sigmoid()
        selected_a = (weights * torch.cat(slopes, dim=1)).sum(dim=1, keepdim=True)
        selected_b = (weights * torch.cat(intercepts, dim=1)).sum(dim=1, keepdim=True)
        base_a = F.interpolate(a, size=guide.shape[-2:], mode="bilinear", align_corners=True)
        base_b = F.interpolate(b, size=guide.shape[-2:], mode="bilinear", align_corners=True)
        return (base_a + blend * (selected_a - base_a),
                base_b + blend * (selected_b - base_b), weights, blend)


class AffineUpdate(nn.Module):
    """One bounded, gated HR re-estimation of both affine coefficients."""

    def __init__(self, update_b: bool = True):
        super().__init__()
        self.update_b = update_b
        self.net = nn.Sequential(nn.Conv2d(10, 16, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(16, 16, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(16, 3, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, guide, up_y, edges_hr, a, b, reconstruction):
        features = torch.cat((guide, up_y, reconstruction, a, b,
                              gradient_magnitude(reconstruction), edges_hr), dim=1)
        raw_a, raw_b, raw_gate = self.net(features).chunk(3, dim=1)
        gate = raw_gate.sigmoid()
        delta_a = 0.2 * raw_a.tanh() * gate
        delta_b = 0.2 * raw_b.tanh() * gate * float(self.update_b)
        return a + delta_a, b + delta_b, delta_a, delta_b


class ExpEAffine(nn.Module):
    def __init__(self, multi_box: bool = True, use_edges: bool = False,
                 hr_fusion: bool = False, stages: int = 0,
                 boxes: tuple[int, ...] = (3, 9), edge_mode: str = "both",
                 update_b: bool = True, affine_estimator: str | None = None,
                 free_b: bool = False, branch_edges: bool = False,
                 route_supervision: bool = False):
        super().__init__()
        self.stages = stages
        self.use_edges = use_edges
        self.branch_edges = branch_edges
        self.route_supervision = route_supervision
        self.hr_fusion = hr_fusion
        self.edge_mode = edge_mode
        self.lr_frozen = False
        self.lr = VIVNet(inp_channels=1, out_channels=1, dim=32,
                         num_blocks=[1, 1, 1], num_refinement_blocks=0,
                         ffn_expansion_factor=3, bias=False)
        if affine_estimator is not None:
            if affine_estimator not in {"learned_stats", "direct"}:
                raise ValueError(f"Unknown affine estimator: {affine_estimator}")
            self.gf = LearnedKernelAffine(direct=affine_estimator == "direct",
                                          use_edges=use_edges, free_b=free_b,
                                          branch_edges=branch_edges,
                                          route_supervision=route_supervision)
        else:
            self.gf = MultiBoxAffine(use_edges, boxes) if multi_box else ConvGuidedFilter()
        self.fusion = HRNeighborFusion() if hr_fusion else None
        self.updates = nn.ModuleList(AffineUpdate(update_b) for _ in range(stages))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.lr_frozen:
            self.lr.eval()
        return self

    def predict_lr(self, lr: torch.Tensor) -> torch.Tensor:
        return run_padded(self.lr, lr - 0.5) + 0.5

    def forward(self, guide: torch.Tensor, lr: torch.Tensor, return_aux: bool = False):
        return self.reconstruct(guide, lr, self.predict_lr(lr), return_aux=return_aux)

    def reconstruct(self, guide: torch.Tensor, lr: torch.Tensor, y_lr: torch.Tensor,
                    return_aux: bool = False):
        if guide.shape[-2:] != (2 * lr.shape[-2], 2 * lr.shape[-1]):
            raise ValueError("Exp E affine models require a 2x HR guide and LR input")
        if y_lr.shape != lr.shape:
            raise ValueError("The supplied LR prediction must match the LR input")
        x_lr, x_hr = lr - 0.5, guide - 0.5
        aux = {"y_lr": y_lr}
        if isinstance(self.gf, ConvGuidedFilter):
            result = self.gf(x_lr, y_lr, x_hr)
            aux["stages"] = [result]
            return (result, aux) if return_aux else result
        needs_edges = self.use_edges or self.branch_edges
        edges_hr, edges_lr = (edge_features(x_hr, y_lr, self.edge_mode)
                              if needs_edges else (None, None))
        a_lr, b_lr, coefficients = self.gf(x_lr, y_lr, edges_lr)
        aux.update(coefficients)
        if self.hr_fusion:
            a, b, weights, blend = self.fusion(a_lr, b_lr, x_lr, y_lr, x_hr, edges_hr)
            aux["hr_weights"] = weights
            aux["hr_blend"] = blend
        else:
            a = F.interpolate(a_lr, size=x_hr.shape[-2:], mode="bilinear", align_corners=True)
            b = F.interpolate(b_lr, size=x_hr.shape[-2:], mode="bilinear", align_corners=True)
        result = (a * x_hr + b).clamp(0, 1)
        aux["stages"] = [result]
        aux["delta_a"], aux["delta_b"] = [], []
        up_y = F.interpolate(y_lr, size=x_hr.shape[-2:], mode="bilinear", align_corners=True)
        for update in self.updates:
            a, b, delta_a, delta_b = update(x_hr, up_y, edges_hr, a, b, result)
            result = (a * x_hr + b).clamp(0, 1)
            aux["stages"].append(result)
            aux["delta_a"].append(delta_a)
            aux["delta_b"].append(delta_b)
        if return_aux:
            aux["a_hr"], aux["b_hr"] = a, b
            return result, aux
        return result


def build_expe_model(variant: str) -> nn.Module:
    if variant == "exp7c_e0":
        return Exp7CE0()
    try:
        return ExpEAffine(**(EXPE_VARIANTS | LEGACY_A_VARIANTS)[variant])
    except KeyError as exc:
        raise ValueError(f"Unknown Exp E variant: {variant}") from exc
