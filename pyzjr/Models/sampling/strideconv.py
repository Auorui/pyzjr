"""
Copyright (c) 2024, Auorui.
All rights reserved.
Time: 2024-01-22 23:14
"""
import torch
import torch.nn as nn

class StridedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, is_relu=True):
        super(StridedConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.is_relu = is_relu

    def forward(self, x):
        x = self.conv(x)
        if self.is_relu:
            x = self.relu(x)
        return x

if __name__ == '__main__':
    input_data = torch.rand((1, 3, 64, 64))
    strided_conv = StridedConv(3, 64)
    output_data = strided_conv(input_data)
    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)
