import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GhostModule(nn.Module):
    """添加了DFC注意力机制,原文使用的是max_pool,这里使用的是avg_pool"""
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, use_attn=True):
        super(GhostModule, self).__init__()
        self.use_attn = use_attn
        self.gate_fn = nn.Sigmoid()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.short_conv = nn.Sequential(
            # horizontal DFC and vertical DFC
            nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(oup),
            nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0,2), groups=oup,bias=False),
            nn.BatchNorm2d(oup),
            nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2,0), groups=oup,bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = out[:, : self.oup, :, :]
        if self.use_attn:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            res = self.short_conv(x)
            return out * F.interpolate(self.gate_fn(res), size=(out.shape[-2], out.shape[-1]), mode='nearest')
        else:
            return out

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x