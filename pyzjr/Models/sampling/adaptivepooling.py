"""
Copyright (c) 2024, Auorui.
All rights reserved.
Time: 2024-01-12 16:14

组合池化, 并且有  ’ 可选择类型 ‘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def adaptive_pool2d(input_tensor, output_size, pool_type='max'):
    """
    两种选择型自适应池化，'max' 和 'avg'

    Args:
        - input_tensor: 输入张量
        - output_size: 输出尺寸
        - pool_type: 池化类型，可以是 'max' 或 'avg'

    Returns:
        - output_tensor: 池化后的张量
    """
    if pool_type == 'max':
        pool_func = F.adaptive_max_pool2d
    elif pool_type == 'avg':
        pool_func = F.adaptive_avg_pool2d
    else:
        raise ValueError("Unsupported pooling type. Use 'max' or 'avg'.")

    output_tensor = pool_func(input_tensor, output_size)
    return output_tensor

def adaptive_avgmax_pool2d(x, output_size):
    """
    两种选择型自适应池化，'max' 和 'avg'的平均值

    Args:
        - x: 输入张量
        - output_size: 输出尺寸

    Returns:
        - 池化后的张量的平均值
    """
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)

def adaptive_catavgmax_pool2d(x, output_size):
    """
    两种选择型自适应池化，'max' 和 'avg'的cat连接

    Args:
        - x: 输入张量
        - output_size: 输出尺寸

    Returns:
        - 连接池化后的张量
    """
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), dim=1)

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
        return adaptive_avgmax_pool2d(x, self.output_size)

class AdaptiveCatAvgMaxPool2d(nn.Module):
    """
    自适应连接平均最大池化的PyTorch模块

    Args:
        - output_size: 输出尺寸
    """
    def __init__(self, output_size=1):
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_catavgmax_pool2d(x, self.output_size)

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
    output_max = adaptive_pool2d(input_tensor, (1, 1), pool_type='max')
    output_avg = adaptive_pool2d(input_tensor, (1, 1), pool_type='avg')
    output_avgmax = adaptive_avgmax_pool2d(input_tensor, (1, 1))
    output_catavgmax = adaptive_catavgmax_pool2d(input_tensor, (1, 1))
    output_fast = SelectAdaptivePool2d()(input_tensor)
    print("自适应最大池化结果：\n", output_max.shape)
    print("\n自适应平均池化结果：\n", output_avg.shape)
    print("\n自适应平均池化与自适应最大池化的结合结果：\n", output_avgmax.shape)
    print("\n自适应平均池化与自适应最大池化的连接结果：\n", output_catavgmax.shape)
    print("\n自定义的自适应平均池化层:\n", output_fast.shape)