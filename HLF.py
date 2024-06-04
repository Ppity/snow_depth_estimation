import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, relu=True):
        super(ConvBnReLU, self).__init__()
        self.add_module(
            'Conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
        )
        self.add_module('BN', nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.999))
        if relu:
            self.add_module('ReLU', nn.ReLU())

class HLF(nn.Module):
    def __init__(self,left,right):
        super(HLF, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                left, left, kernel_size=3, stride=1,
                padding=2, dilation=2),
            nn.BatchNorm2d(left),
            nn.ReLU()
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                left, left, kernel_size=3, stride=1,
                padding=2,dilation=2),
            nn.BatchNorm2d(left),
            nn.ReLU()
        )
        self.right1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                right, right, kernel_size=3, stride=1,
                padding=2,dilation=2),
            nn.BatchNorm2d(right),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
                    ConvBnReLU(640, 256, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x_d, x_s):
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        left2 = F.interpolate(left2, size=16, mode='bilinear', align_corners=True)
        right1 = self.right1(x_s)
        left = left1 * F.sigmoid(right1)
        left = F.interpolate(left, size=16, mode='bilinear', align_corners=True)
        right2 = self.right2(x_s)
        right = left2 * F.sigmoid(right2)
        out = left + right
        out = self.conv(out)
        return out
