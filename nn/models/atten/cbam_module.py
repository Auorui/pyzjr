"""
CBAM (Convolutional Block Attention Module)
Original paper addresshttps: https://arxiv.org/pdf/1807.06521.pdf
Blog records: https://blog.csdn.net/m0_62919535/article/details/136334691
Time: 2024-02-28
"""
import torch
from torch import nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAMAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16, kernel_size=7):
        super(CBAMAttention, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class LightChannelAttention(ChannelAttention):
    """An experimental 'lightweight' that sums avg + max pool first"""
    def __init__(self, channels, reduction=16):
        super(LightChannelAttention, self).__init__(channels, reduction)

    def forward(self, x):
        x_pool = 0.5 * x.mean((2, 3), keepdim=True) + 0.5 * F.adaptive_max_pool2d(x, 1)
        x_attn = self.mlp(x_pool)
        return x * x_attn.sigmoid()

class LightSpatialAttention(nn.Module):
    """An experimental 'lightweight' variant that sums avg_pool and max_pool results."""
    def __init__(self, kernel_size=7, padding=3):
        super(LightSpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_attn = 0.5 * x_avg + 0.5 * max_out
        x_attn = self.conv1(x_attn)
        return x * x_attn.sigmoid()

class LightCBAMAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16, kernel_size=7):
        super(LightCBAMAttention, self).__init__()
        self.channel = LightChannelAttention(in_planes, reduction_ratio)
        self.spatial = LightSpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x

if __name__ == '__main__':
    import torchsummary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(1, 16, 8, 8).to(device)
    net = LightCBAMAttention(16)
    net = net.to(device)
    out = net(input)
    print(out.shape)
    torchsummary.summary(net, input_size=(16, 8, 8))
    # CBAMAttention Total params: 162
    # LightCBAMAttention Trainable params: 81