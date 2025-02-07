"""
Copyright (c) 2024, Auorui.
All rights reserved.

论文原址：https://arxiv.org/pdf/1704.04861.pdf
MobileNet_v1 使用的深度可分离卷积
Depthwise Separable Conv
"""
import math
import torch
import torch.nn as nn

class DWConv(nn.Conv2d):
    # Depth-wise convolution class
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1):
                        # in_channels, out_channels, kernel, stride, padding, groups
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, groups=math.gcd(in_channels, out_channels))

class PWConv(nn.Conv2d):
    # Point-wise convolution class
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

class DepthwiseSeparableConv2d(nn.Module):
    """
    Depth-wise separable convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
            bias=bias,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class DepthwiseSeparableConv2dBlock(nn.Module):
    """
    Depthwise seperable convolution with batchnorm and activation.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            activation=nn.ReLU(inplace=True),
            kernel_size: int = 3,
            stride=1,
            padding=1,
            dilation=1,
    ):
        super(DepthwiseSeparableConv2dBlock, self).__init__()
        self.depthwise = DepthwiseSeparableConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        return self.act(x)


class DepthSepConv(nn.Module):
    """
    深度可分卷积: DW卷积 + PW卷积
    dw卷积, 当分组个数等于输入通道数时, 输出矩阵的通道输也变成了输入通道数
    pw卷积, 使用了1x1的卷积核与普通的卷积一样
    """
    def __init__(self, in_channels, out_channels, stride):
        super(DepthSepConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.batch_norm1(x)
        x = self.relu6(x)
        x = self.pointwise(x)
        x = self.batch_norm2(x)
        x = self.relu6(x)

        return x

if __name__=="__main__":
    input_tensor = torch.randn(1, 3, 32, 32)
    depthconv = DepthSepConv(3, 6, 1)
    output_tensor = depthconv(input_tensor)
    print(output_tensor.shape)
