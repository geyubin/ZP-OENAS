import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(Conv, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if isinstance(kernel_size, int):
            self.ops = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(C_out, affine=True),
            )
        else:
            assert isinstance(kernel_size, tuple)
            k1, k2 = kernel_size[0], kernel_size[1]
            self.ops = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(C_in, C_out, kernel_size=(k1, k2), stride=(1, stride), padding=padding[0], bias=False),
                nn.BatchNorm2d(C_out, affine=True),
                nn.ReLU(inplace=False),
                nn.Conv2d(C_out, C_out, (k2, k1), stride=(stride, 1), padding=padding[1], bias=False),
                nn.BatchNorm2d(C_out, affine=True),
            )

    def forward(self, x):
        x = self.ops(x)
        return x


class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride, affine=True):
        super(MaxPool, self).__init__()
        self.ops = nn.MaxPool2d(kernel_size, stride, padding=1)

    def forward(self, x):
        x = self.ops(x)
        return x


class AvgPool(nn.Module):
    def __init__(self, kernel_size, stride, affine=True):
        super(AvgPool, self).__init__()
        self.ops = nn.AvgPool2d(kernel_size, stride, padding=1)

    def forward(self, x):
        x = self.ops(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel, bias=False),
        #     nn.Sigmoid()
        # )
        self.fc1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.fc1(self.avg_pool(x))
        max_out = self.fc1(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x


Operations_4 = {


    0: lambda c_in, c_out, stride: Conv(c_in, c_out, (1, 3), stride, ((0, 1), (1, 0))),

    1: lambda c_in, c_out, stride: Conv(c_in, c_out, (1, 7), stride, ((0, 3), (3, 0))),

    2: lambda c_in, c_out, stride: Conv(c_in, c_out, (1, 5), stride, ((0, 2), (2, 0))),

    3: lambda c_in, c_out, stride: Conv(c_in, c_out, (1, 1), stride, ((0, 0), (0, 0))),

    4: lambda c_in, c_out, stride: MaxPool(kernel_size=3, stride=stride),

}

Operations_4_name = [
    'Conv 1*1',
    'Conv 3*3',
    'Conv 1*3+3*1',
    'Conv 1*7+7*1',
    'MaxPool 3*3',
]


class FactorizedReduce(nn.Module):  # downsampling
    def __init__(self, C_in, C_out, shape, affine=True):
        super(FactorizedReduce, self).__init__()
        self.shape = shape
        self.C_in = C_in
        self.C_out = C_out

        assert C_out % 2 == 0
        self.path1 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False),
                                   nn.Conv2d(C_in, C_out // 2, 1, bias=False))
        self.path2 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False),
                                   nn.Conv2d(C_in, C_out // 2, 1, bias=False))
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn.train()
        path1 = x
        path2 = F.pad(x, (0, 1, 0, 1), "constant", 0)[:, :, 1:, 1:]
        out = torch.cat([self.path1(path1), self.path2(path2)], dim=1)
        out = self.bn(out)
        return out


class ConvBNReLU(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ConvBNReLU, self).__init__()

        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, bn_train=False):
        x = self.conv(x)
        if bn_train:
            self.bn.train()
        x = self.bn(x)
        x = self.relu(x)
        return x


class MaybeCalibrateSize(nn.Module):
    def __init__(self, layer, channels, affine=True):
        super(MaybeCalibrateSize, self).__init__()
        self.channels = channels
        self.affine = affine
        self.proj_layers = nn.ModuleDict()

    def forward(self, s0, bn_train=False):
        in_channels = s0.size(1)
        if in_channels != self.channels:
            key = str(in_channels)
            if key not in self.proj_layers:
                self.proj_layers[key] = ConvBNReLU(in_channels, self.channels, 1, 1, 0, self.affine).to(s0.device)
            out = self.proj_layers[key](s0, bn_train=bn_train)
        else:
            out = s0
        return out


class AuxHeadCIFAR(nn.Module):
    def __init__(self, C_in, classes):
        """assuming input size 8x8"""
        super(AuxHeadCIFAR, self).__init__()

        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)

        self.classifier = nn.Linear(768, classes)

    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn1.train()
            self.bn2.train()
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

class AuxHeadImageNet(nn.Module):
    def __init__(self, C_in, classes):
        """input should be in [B, C, 7, 7]"""
        super(AuxHeadImageNet, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(768, classes)

    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn1.train()
            self.bn2.train()
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x
