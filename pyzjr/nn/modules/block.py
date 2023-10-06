import torch
import torch.nn as nn
import torch.nn.functional as F

class CBLR(nn.Sequential):
    """
    CBLR -> Conv+BN+LeakyReLU , Originating from darknet53
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, negative_slope=0.1):
        super(CBLR, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope)
        )

class ConvBnAct(nn.Module):
    """
    ConvBnAct -> Conv+Bn+Act(可选)  默认是ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, act_layer=nn.ReLU, inplace=True):
        super(ConvBnAct, self).__init__(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        act_layer(inplace=inplace)
        )

def _make_divisible(v, divisor, min_value=None):
    """
    此函数取自TensorFlow代码库.它确保所有层都有一个可被8整除的通道编号
    在这里可以看到:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    通过四舍五入和增加修正，确保通道编号是可被 divisor 整除的最接近的值，并且保证结果不小于指定的最小值。
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 确保四舍五入的下降幅度不超过10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    """
    实现硬切线函数（hard sigmoid）的函数。
    Args:
        x: 输入张量，可以是任意形状的张量。
        inplace: 是否原地操作（in-place operation）。默认为 False。
    Returns:
        处理后的张量，形状与输入张量相同。
    注意：
        ReLU6 函数是一个将小于 0 的值设为 0，大于 6 的值设为 6 的函数。
        clamp_ 方法用于限制张量的取值范围。
    """
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x