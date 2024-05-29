"""
(CVPR 2020)Strip Pooling: Rethinking Spatial Pooling for Scene Parsing
Original address of the paper: <https://arxiv.org/pdf/2003.13328.pdf>
Code reference: https://github.com/houqb/SPNet
"""
import torch
from torch import nn
import torch.nn.functional as F

class StripPooling(nn.Module):
    def __init__(self, in_channels, pool_size, norm_layer=None):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        inter_channels = int(in_channels / 4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(inplace=True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                     norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                     norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(inplace=True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels * 2, in_channels, 1, bias=False),
                                   norm_layer(in_channels))

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), mode='bilinear', align_corners=True)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), mode='bilinear', align_corners=True)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), mode='bilinear', align_corners=True)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), mode='bilinear', align_corners=True)
        # Feature maps for adaptive average pooling
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        # Feature maps of strip pooling
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)

if __name__ == '__main__':
    block = StripPooling(16, (20, 12))
    input = torch.rand(4, 16, 64, 64)
    output = block(input)
    print(output.shape)