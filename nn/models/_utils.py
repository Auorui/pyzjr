import torch
import math
import numpy as np
import torch.nn as nn

def autopad(kernel, padding=None, dilation=1):
    """自动计算填充大小，以使输出具有与输入相同的形状
    :param k: kernel
    :param p: padding
    :param d: dilation
    :return: 自动计算得到的填充大小
    """
    k, p, d = kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    Args:
        v: The number of input channels.
        divisor: The number of channels should be a multiple of this value.
        min_value: The minimum value of the number of channels, which defaults to the advisor.

    Returns: It ensures that all layers have a channel number that is divisible by 8
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def channel_shuffle(x, groups):
    """Shufflenet uses channel shuffling"""
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)   # reshape
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, num_channels, height, width)  # flatten
    return x

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

if __name__=="__main__":
    input_size = (112, 112, 64)
    kernel_size = 3
    stride = 2
    padding = 3
    conv_output_size = convpool_outsize(input_size, kernel_size, stride, padding, is_conv=True)
    print("卷积输出尺寸:", conv_output_size)
    pool_output_size = convpool_outsize(input_size, kernel_size, stride, padding, is_conv=False)
    print("池化输出尺寸:", pool_output_size)
