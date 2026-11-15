"""
Copyright (c) 2024, Auorui.
All rights reserved.

The Torch implementation of average pooling and maximum pooling has been compared with the official Torch implementation
Time: 2024-01-22  17:28
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyzjr.data.helper import to_2tuple, to_4tuple

__all__ = ["MedianPool2d", "SelectAdaptivePool2d"]


class MedianPool2d(nn.Module):
    """
    Median pool (usable as median filter when stride=1).
    Hacked together by / Copyright 2020 Ross Wightman.
    Currently, it is rarely used in CNN, but extremely effective for noise removal.
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super().__init__()
        self.k = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.padding = to_4tuple(padding)
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            ph = max(self.k[0] - (ih % self.stride[0] if ih % self.stride[0] != 0 else self.stride[0]), 0)
            pw = max(self.k[1] - (iw % self.stride[1] if iw % self.stride[1] != 0 else self.stride[1]), 0)
            pl, pr = pw // 2, pw - pw // 2
            pt, pb = ph // 2, ph - ph // 2
            return pl, pr, pt, pb
        return self.padding

    def forward(self, x):
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class SelectAdaptivePool2d(nn.Module):
    """
    Selectable global pooling layer with dynamic input kernel size.
    """
    def __init__(self, output_size=1, pool_type='fast', flatten=False):
        super().__init__()
        self.pool_type = pool_type
        self.flatten = flatten

        if pool_type == '':
            self.pool = nn.Identity()
        elif pool_type == 'fast':
            assert output_size == 1
            self.pool = lambda x: x.mean((2, 3)) if flatten else x.mean((2, 3), keepdim=True)
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        elif pool_type == 'avgmax':
            self.pool = lambda x: 0.5 * (F.adaptive_avg_pool2d(x, output_size) + F.adaptive_max_pool2d(x, output_size))
        elif pool_type == 'catavgmax':
            self.pool = lambda x: torch.cat((F.adaptive_avg_pool2d(x, output_size),
                                             F.adaptive_max_pool2d(x, output_size)), dim=1)
        else:
            raise ValueError(f'Invalid pool type: {pool_type}')

    def forward(self, x):
        return self.pool(x)