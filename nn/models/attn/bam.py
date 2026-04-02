"""
BAM: Bottleneck Attention Module
Original paper addresshttps: https://arxiv.org/pdf/1807.06514.pdf
The code comes from the official implementation:
https://github.com/Jongchan/attention-module/tree/master
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelGate(nn.Module):
    def __init__(self, dim, reduction=16, num_layers=1):
        super(ChannelGate, self).__init__()
        gate_channels = [dim]
        gate_channels += [dim // reduction] * num_layers
        gate_channels += [dim]
        layers = []
        for i in range(len(gate_channels) - 2):
            layers.extend([
                nn.Linear(gate_channels[i], gate_channels[i+1]),
                nn.BatchNorm1d(gate_channels[i+1]),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.Linear(gate_channels[-2], gate_channels[-1]))
        self.gate_c = nn.Sequential(*layers)

    def forward(self, in_tensor):
        # Apply average pooling
        avg_pool = F.avg_pool2d(in_tensor, in_tensor.size(2), stride=in_tensor.size(2))
        batch_size = avg_pool.size(0)
        flattened = avg_pool.view(batch_size, -1)
        x = self.gate_c(flattened)
        # Reshape back to original spatial dimensions
        return x.unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

class SpatialGate(nn.Module):
    def __init__(self, dim, reduction=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1),
            nn.BatchNorm2d(dim // reduction),
            nn.ReLU(inplace=True),
        )

        self.dilation_blocks = nn.ModuleList()
        for i in range(dilation_conv_num):
            block = nn.Sequential(
                nn.Conv2d(dim // reduction, dim // reduction,
                          kernel_size=3, padding=dilation_val, dilation=dilation_val),
                nn.BatchNorm2d(dim // reduction),
                nn.ReLU(inplace=True)
            )
            self.dilation_blocks.append(block)

        self.conv_final = nn.Conv2d(dim // reduction, 1, kernel_size=1)

    def forward(self, in_tensor):
        x = self.gate_s(in_tensor)
        for block in self.dilation_blocks:
            x = block(x)
        x = self.conv_final(x)
        return x.expand_as(in_tensor)

class BAMAttention(nn.Module):
    def __init__(self, dim, reduction=16):
        super(BAMAttention, self).__init__()
        self.channel_att = ChannelGate(dim, reduction)
        self.spatial_att = SpatialGate(dim, reduction)

    def forward(self,x):
        att = 1 + torch.sigmoid(self.channel_att(x) * self.spatial_att(x))
        return att * x

if __name__ == "__main__":
    input_tensor = torch.randn((2, 64, 32, 32))
    block = BAMAttention(64)
    output_tensor = block(input_tensor)
    print(output_tensor.shape)