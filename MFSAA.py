import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvBnReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, relu=True):
        super(ConvBnReLU, self).__init__()
        self.add_module(
            'Conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
        )
        self.add_module('BN', nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.999))
        if relu:
            self.add_module('ReLU', nn.ReLU())

class ImagePool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, x):
        x_size = x.shape
        x = self.pool(x)
        x = self.conv(x)
        x = F.interpolate(x, size=x_size[2:], mode='bilinear', align_corners=True)
        return x

class MSFAA(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(MSFAA, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module('c0',
                               ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1))
        for idx, rate in enumerate(rates):
            self.stages.add_module('c{}'.format(idx + 1),
                                   ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=rate,
                                              dilation=rate))

        self.stages.add_module('ImagePool', ImagePool(in_channels, out_channels))

    def forward(self, x):
        out = torch.cat([stage(x) for stage in self.stages.children()], dim=1)
        return out