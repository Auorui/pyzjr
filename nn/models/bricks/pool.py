"""
Copyright (c) 2024, Auorui.
All rights reserved.

The Torch implementation of average pooling and maximum pooling has been compared with the official Torch implementation
Time: 2024-01-22  17:28
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyzjr.data.utils.tuplefun import to_2tuple, to_4tuple

class StridedPool2d(nn.Module):
    """
    实现一个跨步卷积层Strided Convolution, 本质上可以实现类似于池化操作的效果。
    通过步幅stride大于 1 的卷积操作来实现空间分辨率的降低。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, is_relu=True):
        super(StridedPool2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.is_relu = is_relu

    def forward(self, x):
        x = self.conv(x)
        if self.is_relu:
            x = self.relu(x)
        return x

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    Hacked together by / Copyright 2020 Ross Wightman
    Currently, it is rarely used in CNN, and the code is taken
    from the project pytorch-image-models

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.padding = to_4tuple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

class AdaptiveAvgMaxPool2d(nn.Module):
    """
    自适应平均最大池化的PyTorch模块

    Args:
        - output_size: 输出尺寸
    """
    def __init__(self, output_size=1):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        x_avg = F.adaptive_avg_pool2d(x, self.output_size)
        x_max = F.adaptive_max_pool2d(x, self.output_size)
        return 0.5 * (x_avg + x_max)

class AdaptiveCatAvgMaxPool2d(nn.Module):
    """
    拼接自适应平均最大池化的PyTorch模块

    Args:
        - output_size: 输出尺寸
    """
    def __init__(self, output_size=1):
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        x_avg = F.adaptive_avg_pool2d(x, self.output_size)
        x_max = F.adaptive_max_pool2d(x, self.output_size)
        return torch.cat((x_avg, x_max), dim=1)

class FastAdaptiveAvgPool2d(nn.Module):
    """
    自定义的自适应平均池化层

    Args:
        - flatten: 是否对结果进行扁平化
    """
    def __init__(self, flatten=False):
        super(FastAdaptiveAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        return x.mean((2, 3)) if self.flatten else x.mean((2, 3), keepdim=True)


class SelectAdaptivePool2d(nn.Module):
    """
    Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size=1, pool_type='fast', flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        self.flatten = flatten
        if pool_type == '':
            self.pool = nn.Identity()  # pass through
        elif pool_type == 'fast':
            assert output_size == 1
            self.pool = FastAdaptiveAvgPool2d(self.flatten)
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        elif pool_type == 'avgmax':
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == 'catavgmax':
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        else:
            assert False, 'Invalid pool type: %s' % pool_type

    def forward(self, x):
        x = self.pool(x)
        return x


if __name__ == "__main__":
    input_tensor = torch.rand((1, 3, 32, 32))
    output_max = F.adaptive_max_pool2d(input_tensor, (1, 1))
    output_avg = F.adaptive_avg_pool2d(input_tensor, (1, 1))
    output_avgmax = AdaptiveAvgMaxPool2d(1)(input_tensor)
    output_catavgmax = AdaptiveCatAvgMaxPool2d(1)(input_tensor)
    output_fast = SelectAdaptivePool2d()(input_tensor)
    print("自适应最大池化结果：\n", output_max.shape)
    print("\n自适应平均池化结果：\n", output_avg.shape)
    print("\n自适应平均池化与自适应最大池化的结合结果：\n", output_avgmax.shape)
    print("\n自适应平均池化与自适应最大池化的连接结果：\n", output_catavgmax.shape)
    print("\n自定义的自适应平均池化层:\n", output_fast.shape)

    # MaxPool2d与AvgPool2d手写测试实验
    input_data = torch.rand((1, 3, 3, 3))
    # input_data = torch.Tensor([[[[0.3939, 0.8964, 0.3681],
    #                            [0.5134, 0.3780, 0.0047],
    #                            [0.0681, 0.0989, 0.5962]],
    #                           [[0.7954, 0.4811, 0.3329],
    #                            [0.8804, 0.3986, 0.3561],
    #                            [0.2797, 0.3672, 0.6508]],
    #                           [[0.6309, 0.1340, 0.0564],
    #                            [0.3101, 0.9927, 0.5554],
    #                            [0.0947, 0.2305, 0.8299]]]])

    print(input_data.shape)
    is_vectoring = False
    kernel_size = 3
    stride = 2
    MaxPool2d1 = nn.MaxPool2d(kernel_size, stride)
    output_data_with_torch_max = MaxPool2d1(input_data)
    AvgPool2d1 = nn.AvgPool2d(kernel_size, stride)

    print("\ntorch.nn pooling Output:")
    print(output_data_with_torch_max,"\n",output_data_with_torch_max.size())
    # 直接使用bool方法判断会因为浮点数的原因出现偏差

    # tensor([[[[0.8964]],       # output_data_with_max
    #          [[0.8804]],
    #          [[0.9927]]]])
    # tensor([[[[0.3686]],       # output_data_with_avg
    #           [[0.5047]],
    #           [[0.4261]]]])