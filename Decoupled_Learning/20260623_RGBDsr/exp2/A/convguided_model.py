import torch
from torch import nn
from torch.nn import functional as F


class ConvGuidedFilter(nn.Module):
    """Same-resolution convolutional guided filter used by the DGF baseline."""

    def __init__(self, in_channels=1, num_features=64, radius=1, norm=nn.BatchNorm2d):
        super().__init__()
        self.box_filter = nn.Conv2d(in_channels, in_channels, 3, padding=radius,
                                    dilation=radius, bias=False)
        self.conv_a = nn.Sequential(
            nn.Conv2d(2 * in_channels, num_features, 1, bias=False), norm(num_features),
            nn.ReLU(inplace=True), nn.Conv2d(num_features, num_features, 1, bias=False),
            norm(num_features), nn.ReLU(inplace=True),
            nn.Conv2d(num_features, in_channels, 1, bias=False))
        self.box_filter.weight.data.fill_(1.0)

    def forward(self, guide, depth):
        if depth.shape != guide.shape:
            raise ValueError(f"guide and depth must have the same shape, got {guide.shape} and {depth.shape}")
        _, _, height, width = guide.shape
        normalizer = self.box_filter(guide.new_ones((1, 1, height, width)))
        mean_guide = self.box_filter(guide) / normalizer
        mean_depth = self.box_filter(depth) / normalizer
        covariance = self.box_filter(guide * depth) / normalizer - mean_guide * mean_depth
        variance = self.box_filter(guide * guide) / normalizer - mean_guide * mean_guide
        a = self.conv_a(torch.cat([covariance, variance], dim=1))
        b = mean_depth - a * mean_guide
        mean_a = self.box_filter(a) / normalizer
        mean_b = self.box_filter(b) / normalizer
        return mean_a * guide + mean_b


class SurveyDGF(nn.Module):
    """Grayscale RGB guide with a learnable, initially near-zero blur blend."""

    def __init__(self, num_features=64):
        super().__init__()
        self.guide_smoothing_logit = nn.Parameter(torch.tensor(-4.0))
        self.gf = ConvGuidedFilter(in_channels=1, num_features=num_features)

    def _build_guide(self, guide_raw):
        padded = F.pad(guide_raw, (1, 1, 1, 1), mode="reflect")
        guide_blur = F.avg_pool2d(padded, kernel_size=3, stride=1)
        alpha = torch.sigmoid(self.guide_smoothing_logit)
        return (1 - alpha) * guide_raw + alpha * guide_blur

    def forward(self, depth_lr, rgb_hr):
        guide_raw = rgb_hr.mean(dim=1, keepdim=True)
        depth_up = F.interpolate(depth_lr, size=guide_raw.shape[-2:], mode="bicubic", align_corners=False)
        return self.gf(self._build_guide(guide_raw), depth_up)
