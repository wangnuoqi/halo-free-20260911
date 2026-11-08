"""Four interpretable, single-channel WLS restoration hypotheses for Exp C."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ChannelNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + 1e-6) * self.weight + self.bias


class NAFBlock(nn.Module):
    """Small NAF-style block: gated depthwise mixing and channel attention."""
    def __init__(self, channels: int):
        super().__init__()
        hidden = channels * 2
        self.norm1 = ChannelNorm(channels)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.project = nn.Conv2d(channels, channels, 1)
        self.norm2 = ChannelNorm(channels)
        self.ffn1 = nn.Conv2d(channels, hidden, 1)
        self.ffn2 = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        z = self.depthwise(self.expand(self.norm1(x)))
        z1, z2 = z.chunk(2, dim=1)
        z = self.project((z1 * z2) * self.sca(z1 * z2))
        x = x + self.beta * z
        z = self.ffn1(self.norm2(x))
        z1, z2 = z.chunk(2, dim=1)
        return x + self.gamma * self.ffn2(z1 * z2)


class NAFSmall(nn.Module):
    """C1: narrow U-shaped NAF baseline with a global residual output."""
    def __init__(self):
        super().__init__()
        self.intro = nn.Conv2d(1, 24, 3, padding=1)
        self.enc0 = nn.Sequential(NAFBlock(24), NAFBlock(24))
        self.down = nn.Conv2d(24, 48, 2, stride=2)
        self.enc1 = nn.Sequential(NAFBlock(48), NAFBlock(48), NAFBlock(48))
        self.mid = nn.Sequential(NAFBlock(48), NAFBlock(48), NAFBlock(48), NAFBlock(48))
        self.up = nn.Conv2d(48, 24, 1)
        self.fuse = nn.Conv2d(48, 24, 1)
        self.dec = nn.Sequential(NAFBlock(24), NAFBlock(24))
        self.ending = nn.Conv2d(24, 1, 3, padding=1)

    def forward(self, x):
        s = self.enc0(self.intro(x))
        z = self.mid(self.enc1(self.down(s)))
        z = F.interpolate(z, size=s.shape[-2:], mode="bilinear", align_corners=False)
        z = self.dec(self.fuse(torch.cat((s, self.up(z)), dim=1)))
        return x + self.ending(z)


def _unit(x):
    return F.normalize(x, dim=1, eps=1e-6)


class SimilarityFusion(nn.Module):
    """Fuse a skip only where normalized cross-scale features agree."""
    def __init__(self, channels: int):
        super().__init__()
        self.deep_projection = nn.Conv2d(channels * 2, channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2 + 2, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, skip, deep, guide):
        deep = F.interpolate(deep, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        deep = self.deep_projection(deep)
        similarity = (_unit(skip) * _unit(deep)).sum(dim=1, keepdim=True)
        guide = F.interpolate(guide, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.gate(torch.cat((skip, deep, similarity, guide), dim=1))
        return self.fuse(torch.cat((skip * gate, deep), dim=1)), gate


class SimilarityPyramid(nn.Module):
    """C2: a two-scale restoration network with explicit similarity gates."""
    def __init__(self):
        super().__init__()
        self.intro = nn.Conv2d(1, 24, 3, padding=1)
        self.enc0 = nn.Sequential(NAFBlock(24), NAFBlock(24))
        self.down = nn.Conv2d(24, 48, 2, stride=2)
        self.enc1 = nn.Sequential(NAFBlock(48), NAFBlock(48), NAFBlock(48))
        self.mid = nn.Sequential(NAFBlock(48), NAFBlock(48))
        self.skip_fusion = SimilarityFusion(24)
        self.dec = nn.Sequential(NAFBlock(24), NAFBlock(24))
        self.ending = nn.Conv2d(24, 1, 3, padding=1)

    def forward(self, x, return_aux=False):
        guide = torch.abs(x - F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, stride=1))
        s = self.enc0(self.intro(x))
        deep = self.mid(self.enc1(self.down(s)))
        fused, gate = self.skip_fusion(s, deep, guide)
        result = x + self.ending(self.dec(fused))
        if return_aux:
            return result, {"similarity_gate": gate, "edge_guide": guide}
        return result


class WLSNAF(nn.Module):
    """Three-scale residual surrogate with similarity-aware skip selection.

    The network predicts the WLS correction in one encoder-decoder pass.  A
    shallow full-resolution head receives an input high-pass guide so edge
    correction remains explicit without introducing a serial refine stage.
    """
    def __init__(self):
        super().__init__()
        self.intro = nn.Conv2d(1, 32, 3, padding=1)
        self.enc0 = nn.Sequential(NAFBlock(32), NAFBlock(32))
        self.down1 = nn.Conv2d(32, 64, 2, stride=2)
        self.enc1 = nn.Sequential(NAFBlock(64), NAFBlock(64), NAFBlock(64))
        self.down2 = nn.Conv2d(64, 96, 2, stride=2)
        self.mid = nn.Sequential(NAFBlock(96), NAFBlock(96), NAFBlock(96), NAFBlock(96))
        self.up1 = nn.Conv2d(96, 128, 1)
        self.fuse1 = SimilarityFusion(64)
        self.dec1 = nn.Sequential(NAFBlock(64), NAFBlock(64))
        self.up0 = nn.Conv2d(64, 64, 1)
        self.fuse0 = SimilarityFusion(32)
        self.dec0 = nn.Sequential(NAFBlock(32), NAFBlock(32))
        self.edge_head = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GELU(),
            NAFBlock(16),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        self.ending = nn.Conv2d(32, 1, 3, padding=1)
        self.edge_gate = nn.Sequential(
            nn.Conv2d(2, 16, 1), nn.GELU(), nn.Conv2d(16, 1, 1), nn.Sigmoid()
        )

    @staticmethod
    def _guide(x):
        blur = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
        return (x - blur).abs()

    def forward(self, x, return_aux=False):
        guide = self._guide(x)
        s0 = self.enc0(self.intro(x))
        s1 = self.enc1(self.down1(s0))
        deep = self.mid(self.down2(s1))
        f1, gate1 = self.fuse1(s1, self.up1(deep), guide)
        f1 = self.dec1(f1)
        f0, gate0 = self.fuse0(s0, self.up0(f1), guide)
        features = self.dec0(f0)
        base_delta = self.ending(features)
        edge_delta = self.edge_head(torch.cat((x, guide, base_delta), dim=1))
        gate = self.edge_gate(torch.cat((guide, base_delta.abs()), dim=1))
        result = x + base_delta + gate * edge_delta
        if return_aux:
            return result, {"similarity_gate_half": gate1, "similarity_gate_full": gate0,
                            "edge_guide": guide, "edge_gate": gate,
                            "base_delta": base_delta, "edge_delta": edge_delta}
        return result


class LowResolutionBase(nn.Module):
    """C3: predict a smooth base at 1/4 scale, then correct edges at full scale."""
    def __init__(self):
        super().__init__()
        self.base_in = nn.Conv2d(1, 32, 3, padding=1)
        self.base_net = nn.Sequential(NAFBlock(32), NAFBlock(32), NAFBlock(32), NAFBlock(32))
        self.base_out = nn.Conv2d(32, 1, 3, padding=1)
        self.detail_in = nn.Conv2d(3, 16, 3, padding=1)
        self.detail_net = nn.Sequential(NAFBlock(16), NAFBlock(16))
        self.detail_out = nn.Conv2d(16, 2, 3, padding=1)

    def forward(self, x, return_aux=False):
        low = F.interpolate(x, scale_factor=0.25, mode="area")
        base_low = low + self.base_out(self.base_net(self.base_in(low)))
        base = F.interpolate(base_low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        blur = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
        high = x - blur
        correction, gate = self.detail_out(self.detail_net(self.detail_in(torch.cat((x, base, high), dim=1)))).chunk(2, dim=1)
        result = base + torch.sigmoid(gate) * correction
        if return_aux:
            return result, {"base": base, "high_frequency": high, "edge_gate": torch.sigmoid(gate)}
        return result


class AdaptiveNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.ones(1))
        self.w1 = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(channels, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w0 * x + self.w1 * self.bn(x)


class DeepVIVBlock(nn.Module):
    """VIV-inspired residual block used to isolate LR-predictor depth."""
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        hidden = channels * 2
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.local = nn.Conv2d(channels, channels, 3, padding=dilation,
                               dilation=dilation, groups=channels)
        self.wide = nn.Conv2d(channels, channels, 9, padding=4, groups=channels)
        self.order3 = nn.Conv2d(channels, channels, 3, padding=dilation,
                                dilation=dilation, groups=channels)
        self.order4 = nn.Conv2d(channels, channels, 3, padding=dilation,
                                dilation=dilation, groups=channels)
        self.similarity = nn.Sequential(
            nn.Linear(hidden, max(4, channels // 2)), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(max(4, channels // 2), hidden),
        )
        self.project = nn.Conv2d(channels, channels, 1)
        self.norm = AdaptiveNorm(channels)
        self.activation = nn.LeakyReLU(0.2, inplace=False)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        f3, f9 = self.expand(x).chunk(2, dim=1)
        f3, f9 = self.local(f3), self.wide(f9)
        features = F.normalize(torch.cat((f3, f9), dim=1).flatten(2), dim=2, eps=1e-6)
        correlation = torch.bmm(features, features.transpose(1, 2)).mean(dim=2)
        weights = torch.tanh(self.similarity(correlation)).unsqueeze(-1).unsqueeze(-1)
        w3, w9 = weights.chunk(2, dim=1)
        f3, f9 = f3 * (1.0 + w3), f9 * (1.0 + w9)
        mixed = f3 * f9
        mixed = self.order3(mixed) * f9
        mixed = self.order4(mixed) * f3
        return self.activation(self.norm(x + self.project(mixed)))


class DeepGuidedLR(nn.Module):
    """A deeper LR predictor followed by the learned local affine guided filter."""
    def __init__(self, depth: int):
        super().__init__()
        channels = 24
        self.depth = depth
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=False),
            AdaptiveNorm(channels), nn.LeakyReLU(0.2, inplace=False),
        )
        dilations = (2, 4, 8, 16)
        self.blocks = nn.Sequential(*(
            DeepVIVBlock(channels, dilations[index % len(dilations)])
            for index in range(depth)
        ))
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            AdaptiveNorm(channels), nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(channels, 1, 1),
        )
        self.box = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        nn.init.ones_(self.box.weight)
        self.coefficient = nn.Sequential(
            nn.Conv2d(2, 32, 1, bias=False), AdaptiveNorm(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 1, bias=False), AdaptiveNorm(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, 1, 1, bias=False),
        )

    def predict_lr(self, lr):
        return self.tail(self.blocks(self.stem(lr)))

    def forward(self, guide, lr):
        if guide.shape[-2] != lr.shape[-2] * 2 or guide.shape[-1] != lr.shape[-1] * 2:
            raise ValueError("DeepGuidedLR expects a 2x HR guide and LR input")
        y_lr = self.predict_lr(lr)
        count = self.box(torch.ones_like(lr))
        mean_x = self.box(lr) / count
        mean_y = self.box(y_lr) / count
        covariance = self.box(lr * y_lr) / count - mean_x * mean_y
        variance = self.box(lr.square()) / count - mean_x.square()
        a = self.coefficient(torch.cat((covariance, variance), dim=1))
        b = mean_y - a * mean_x
        size = guide.shape[-2:]
        a = F.interpolate(a, size, mode="bilinear", align_corners=True)
        b = F.interpolate(b, size, mode="bilinear", align_corners=True)
        return (a * guide + b).clamp(0.0, 1.0)


class DeepGuidedLR4(DeepGuidedLR):
    def __init__(self):
        super().__init__(depth=4)


class DeepGuidedLR8(DeepGuidedLR):
    def __init__(self):
        super().__init__(depth=8)


class GuidedPyramidLR4(DeepGuidedLR):
    """Route the same four LR blocks through a two-level encoder-decoder."""
    def __init__(self):
        super().__init__(depth=4)

    def predict_lr(self, lr):
        shallow = self.blocks[0](self.stem(lr))
        middle = self.blocks[1](F.avg_pool2d(shallow, 2))
        deep = self.blocks[2](F.avg_pool2d(middle, 2))
        middle = 0.5 * (middle + F.interpolate(
            deep, size=middle.shape[-2:], mode="bilinear", align_corners=False
        ))
        shallow = 0.5 * (shallow + F.interpolate(
            middle, size=shallow.shape[-2:], mode="bilinear", align_corners=False
        ))
        return self.tail(self.blocks[3](shallow))


class EdgeDiffusion(nn.Module):
    """C4: learned edge conductances with a fixed, inspectable diffusion solver."""
    def __init__(self, iterations: int = 6):
        super().__init__()
        self.iterations = iterations
        self.coefficients = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GELU(),
            NAFBlock(16),
            nn.Conv2d(16, 3, 3, padding=1),
        )
        nn.init.zeros_(self.coefficients[-1].weight)
        nn.init.zeros_(self.coefficients[-1].bias)
        self.log_lambda = nn.Parameter(torch.tensor(-0.2))

    @staticmethod
    def _neighbors(x):
        p = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return p[..., 1:-1, 2:], p[..., 1:-1, :-2], p[..., 2:, 1:-1], p[..., :-2, 1:-1]

    def forward(self, x, return_aux=False):
        coefficients = self.coefficients(x)
        base, horizontal, vertical = coefficients.chunk(3, dim=1)
        data = x + 0.1 * torch.tanh(base)
        weight_x = torch.sigmoid(horizontal)
        weight_y = torch.sigmoid(vertical)
        lam = F.softplus(self.log_lambda).clamp(max=2.0)
        value = x
        for _ in range(self.iterations):
            right, left, down, up = self._neighbors(value)
            wx_right = F.pad(weight_x[..., :-1], (0, 1, 0, 0))
            wx_left = F.pad(weight_x[..., :-1], (1, 0, 0, 0))
            wy_down = F.pad(weight_y[..., :-1, :], (0, 0, 0, 1))
            wy_up = F.pad(weight_y[..., :-1, :], (0, 0, 1, 0))
            numerator = data + lam * (
                wx_right * right + wx_left * left + wy_down * down + wy_up * up
            )
            denominator = 1.0 + lam * (wx_right + wx_left + wy_down + wy_up)
            value = numerator / denominator
        if return_aux:
            return value, {"data_term": data, "horizontal_conductance": weight_x,
                           "vertical_conductance": weight_y, "lambda": lam.detach(),
                           "iterations": self.iterations}
        return value


from functools import partial

from expe.model import EXPE_VARIANTS, build_expe_model


VARIANTS = {
    "naf_small": NAFSmall,
    "similarity_pyramid": SimilarityPyramid,
    "wls_naf": WLSNAF,
    "lowres_edge": LowResolutionBase,
    "unrolled_diffusion": EdgeDiffusion,
    "deep_guided_lr4": DeepGuidedLR4,
    "deep_guided_lr8": DeepGuidedLR8,
    "guided_pyramid_lr4": GuidedPyramidLR4,
}
VARIANTS.update({name: partial(build_expe_model, name) for name in EXPE_VARIANTS})

GUIDED_VARIANTS = {"deep_guided_lr4", "deep_guided_lr8", "guided_pyramid_lr4", *EXPE_VARIANTS}


def build_model(variant: str) -> nn.Module:
    try:
        return VARIANTS[variant]()
    except KeyError as exc:
        raise ValueError(f"Unknown Exp C model {variant!r}; choices: {sorted(VARIANTS)}") from exc
