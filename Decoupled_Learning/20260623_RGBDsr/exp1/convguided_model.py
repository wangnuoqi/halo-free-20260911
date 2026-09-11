import torch
from torch import nn
from torch.nn import functional as F


class ConvGuidedFilter(nn.Module):
    """The same-resolution ConvGuidedFilter used by the survey DGF baseline."""

    def __init__(self, in_channels=1, num_features=64, radius=1, norm=nn.BatchNorm2d):
        super().__init__()
        self.box_filter = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=radius,
            dilation=radius,
            bias=False,
        )
        self.conv_a = nn.Sequential(
            nn.Conv2d(2 * in_channels, num_features, kernel_size=1, bias=False),
            norm(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=1, bias=False),
            norm(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, in_channels, kernel_size=1, bias=False),
        )
        self.box_filter.weight.data.fill_(1.0)

    def forward(self, guide, depth):
        _, _, height, width = guide.shape
        if depth.shape != guide.shape:
            raise ValueError(
                f"guide and depth must have the same shape, got {guide.shape} and {depth.shape}"
            )

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
    """Survey DGF: HR grayscale RGB guided by bicubic-upsampled LR depth."""

    def __init__(self, num_features=64):
        super().__init__()
        self.gf = ConvGuidedFilter(in_channels=1, num_features=num_features)

    def forward(self, depth_lr, rgb_hr):
        guide_hr = rgb_hr.mean(dim=1, keepdim=True)
        depth_lr_up = F.interpolate(
            depth_lr,
            size=guide_hr.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        return self.gf(guide_hr, depth_lr_up)
