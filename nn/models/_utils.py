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

def convpool_outsize(input_size, kernel_size, stride, padding, is_conv=True, ceil_mode=False):
    """
    计算卷积或池化的输出尺寸。
    Parameters:
        input_size (tuple): 输入尺寸，格式为 (height, width, channels)
        kernel_size (int or tuple): 卷积核或池化核的尺寸，如果是 int，表示高宽相同的正方形核
        stride (int): 步幅
        padding (int): 填充大小
        is_conv (bool): 是否是卷积操作，如果是 True，使用卷积输出尺寸的计算公式；否则使用池化输出尺寸的计算公式

    Returns:
        output_size (tuple): 输出尺寸，格式为 (height, width, channels)。
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    h_in, w_in, c = input_size
    h_k, w_k = kernel_size

    if is_conv:
        h_out = ((h_in + 2 * padding - h_k) // stride) + 1
        w_out = ((w_in + 2 * padding - w_k) // stride) + 1
    else:
        h_out = ((h_in - h_k) // stride) + 1
        w_out = ((w_in - w_k) // stride) + 1
    if ceil_mode:
        h_out, w_out = math.ceil(h_out), math.ceil(w_out)
    return h_out, w_out, c

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

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
