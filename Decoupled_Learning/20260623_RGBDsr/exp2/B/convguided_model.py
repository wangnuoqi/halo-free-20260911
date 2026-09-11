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
    """RGB/depth gradient confidence blend followed by the baseline guided filter."""

    def __init__(self, num_features=64):
        super().__init__()
        self.confidence_head = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid())
        nn.init.constant_(self.confidence_head[2].bias, 2.0)
        self.gf = ConvGuidedFilter(in_channels=1, num_features=num_features)
        self.last_confidence = None

    @staticmethod
    def _gradient_magnitude(image):
        sobel_x = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        sobel_y = image.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        gx = F.conv2d(F.pad(image, (1, 1, 1, 1), mode="reflect"), sobel_x)
        gy = F.conv2d(F.pad(image, (1, 1, 1, 1), mode="reflect"), sobel_y)
        return torch.sqrt(gx.square() + gy.square() + 1e-12)

    def _build_guide(self, rgb_gray, depth_up):
        features = torch.cat([rgb_gray, depth_up,
                              self._gradient_magnitude(rgb_gray),
                              self._gradient_magnitude(depth_up)], dim=1)
        confidence = self.confidence_head(features)
        return confidence * rgb_gray + (1 - confidence) * depth_up, confidence

    def forward(self, depth_lr, rgb_hr):
        rgb_gray = rgb_hr.mean(dim=1, keepdim=True)
        depth_up = F.interpolate(depth_lr, size=rgb_gray.shape[-2:], mode="bicubic", align_corners=False)
        guide, confidence = self._build_guide(rgb_gray, depth_up)
        self.last_confidence = confidence
        return self.gf(guide, depth_up)
