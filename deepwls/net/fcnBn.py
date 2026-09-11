import torch
from torch import nn
# from util.tool import img_gradient

class ResBlock(nn.Module):
    """
    构建残差块
    """
    def __init__(self, ch_in, ch_out, padding, dilation=1, stride=1):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, (3, 3), stride, padding, dilation),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(True),
            nn.Conv2d(ch_in, ch_out, (3, 3), stride, padding, dilation),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out + x
        return out


class FcnBn(nn.Module):
    """
    构建网络结构
    """
    def __init__(self):
        super(FcnBn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1),  # 输出的通道数，取决于卷积核的数量，就是64嘛？
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.layer4 = ResBlock(64, 64, 4, 4)
        self.layer6 = ResBlock(64, 64, 16, 16)
        self.layer7 = ResBlock(64, 64, 1)
        self.layer8 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (3, 3), 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(64, 3, (3, 3), 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # [b, 3, h, w] => [b, 64, h, w]
        out = self.layer2(self.layer1(x))
        out = self.layer7(self.layer6((self.layer4(out))))
        out = self.layer9(self.layer8(out))
        return out


if __name__ == "__main__":
    from torchinfo import summary

    a = torch.randn((1, 3, 256, 256))
    m = FcnBn()
    b = m(a)
    summary(m, (1, 3, 256, 256))
    print(b.shape)
