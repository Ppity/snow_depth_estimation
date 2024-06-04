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


def convBnReluPool(input_dim, output_dim, stride):
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, padding=0),
        nn.BatchNorm2d(output_dim),
        nn.ReLU(inplace=False),
        # nn.MaxPool2d(kernel_size=2)
    )

def adpLinear():
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 64),
        nn.Dropout(0.1),
        nn.Linear(64, 32),
        nn.Dropout(0.1),
        nn.Linear(32, 1)

    )


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, stride, expansion):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(output_dim)
        self.short_conv = nn.Sequential()
        if stride != 1 and input_dim != expansion * input_dim:  # expansion=2,output_channels=128,num_blocks=2,stride=2
            self.short_conv = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(output_dim)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)),inplace=False)
        out = F.relu(self.bn2(self.conv2(out)),inplace=False)
        out = out + self.short_conv(x)
        out = F.relu(out, inplace=False)
        return out


class MFPANet(nn.Module):
    def __init__(self, block):
        super(MFPANet, self).__init__()
        self.input_channels = 32
        self.convBnRelu_Pool = convBnReluPool(8, self.input_channels, 1)
        self.features_1 = []
        self.features_2 = []
        self.features_3 = []
        self.features_4 = []
        self.features_6 = []
        self.features_center = []
        self.features_inner = []
        self.pool_1 = nn.MaxPool2d(kernel_size=2)
        self.pool_2 = nn.AvgPool2d(kernel_size=2)
        self.pool_3 = nn.AvgPool2d(kernel_size=2)
        self.conv1_1 = ConvBnReLU(32,32,kernel_size=3,stride=2,padding=1)
        self.conv1_2 = ConvBnReLU(32,32,kernel_size=3,stride=1,padding=1)
        self.conv2_1 = ConvBnReLU(32,32, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = ConvBnReLU(32,32, kernel_size=3, stride=2, padding=1)
        self.conv1x1 = ConvBnReLU(32,64,kernel_size=1,stride=1,padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.msfaa_1 = msfaa(64,128)
        self.msfaa_2 = msfaa(64,128)
        self.msfaa_3 = msfaa(128,128)
        self.msfaa_4 = msfaa(128,128)
        self.msfaa_6 = msfaa(128,128)
        self.convbnrelu_1 = ConvBnReLU(256, 128, kernel_size=1, stride=1, padding=0)

        self.input_channels = 32
        block_settings_1 = [
            [1, 32, 2, 1],
            [2, 64, 2, 2],
        ]
        for t, c, n, s in block_settings_1:
            output_channels = c
            for i in range(n):
                if i == 0:
                    self.features_1.append(
                        block(input_dim=self.input_channels, output_dim=output_channels, stride=s, expansion=t))
                else:
                    self.features_1.append(
                        block(input_dim=output_channels, output_dim=output_channels, stride=1, expansion=t))
                self.input_channels = output_channels

        self.input_channels = 32
        block_settings_2 = [
            [2, 64, 2, 2],
            [1, 64, 2, 1],
        ]
        for t, c, n, s in block_settings_2:
            output_channels = c
            for i in range(n):
                if i == 0:
                    self.features_2.append(
                        block(input_dim=self.input_channels, output_dim=output_channels, stride=s, expansion=t))
                else:
                    self.features_2.append(
                        block(input_dim=output_channels, output_dim=output_channels, stride=1, expansion=t))
                self.input_channels = output_channels

        self.input_channels = 64
        block_settings_3 = [
            [1, 64, 2, 1],
            [2, 128, 2, 2],
        ]
        for t, c, n, s in block_settings_3:
            output_channels = c
            for i in range(n):
                if i == 0:
                    self.features_3.append(
                        block(input_dim=self.input_channels, output_dim=output_channels, stride=s, expansion=t))
                else:
                    self.features_3.append(
                        block(input_dim=output_channels, output_dim=output_channels, stride=1, expansion=t))
                self.input_channels = output_channels

        self.input_channels = 64
        block_settings_4 = [
            [2, 128, 2, 2],
            [1, 128, 2, 1],
        ]
        for t, c, n, s in block_settings_4:
            output_channels = c
            for i in range(n):
                if i == 0:
                    self.features_4.append(
                        block(input_dim=self.input_channels, output_dim=output_channels, stride=s, expansion=t))
                else:
                    self.features_4.append(
                        block(input_dim=output_channels, output_dim=output_channels, stride=1, expansion=t))
                self.input_channels = output_channels

        self.input_channels = 64
        block_settings_5 = [
            [2, 128, 2, 2],
            [1, 128, 2, 1],
        ]
        for t, c, n, s in block_settings_5:
            output_channels = c
            for i in range(n):
                if i == 0:
                    self.features_center.append(
                        block(input_dim=self.input_channels, output_dim=output_channels, stride=s, expansion=t))
                else:
                    self.features_center.append(
                        block(input_dim=output_channels, output_dim=output_channels, stride=1, expansion=t))
                self.input_channels = output_channels

        self.input_channels = 128
        block_settings_6 = [
            [2, 256, 2, 2],
            [1, 256, 2, 1],
        ]
        for t, c, n, s in block_settings_6:
            output_channels = c
            for i in range(n):
                if i == 0:
                    self.features_inner.append(
                        block(input_dim=self.input_channels, output_dim=output_channels, stride=s, expansion=t))
                else:
                    self.features_inner.append(
                        block(input_dim=output_channels, output_dim=output_channels, stride=1, expansion=t))
                self.input_channels = output_channels



        self.Linear = adpLinear()
        self.features_1 = nn.Sequential(*self.features_1)
        self.features_2 = nn.Sequential(*self.features_2)
        self.features_3 = nn.Sequential(*self.features_3)
        self.features_4 = nn.Sequential(*self.features_4)
        self.features_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.features_inner.append(ConvBnReLU(256,128,kernel_size=1,stride=1,padding=0))
        self.features_inner = nn.Sequential(*self.features_inner)
        self.a = nn.Parameter(torch.randn(1, requires_grad=True))
        self.cat = bffnet(640,640)
        self.short = nn.Sequential(
            ConvBnReLU(256,256,kernel_size=3,stride=1,padding=4,dilation=4),
            ConvBnReLU(256,256,kernel_size=3,stride=1,padding=4,dilation=4))

    def forward(self, x):
        x = self.convBnRelu_Pool(x)
        out_ = x
        out_1 = self.features_1(out_)
        out_2 = self.features_2(out_)
        out_inner = torch.cat((out_1,out_2),dim=1)
        out_a = self.msfaa_1(out_1)  
        out_b = self.msfaa_2(out_2)    
        out_3 = self.features_3(out_1)
        out_4 = self.features_4(out_2)
        out_c = self.msfaa_3(out_3)  
        out_d = self.msfaa_4(out_4)   
        out_inner = self.features_inner(out_inner)
        out_inner = self.msfaa_6(out_inner)
        outh = out_c + out_d + out_inner  
        outl = out_a + out_b # 640,16,16
        out = self.cat(outh,outl)
        out = self.short(out) + out
        out = self.convbnrelu_1(out)
        out = self.Linear(out)

        return out
def net():
    return MFPANet(BasicBlock)
