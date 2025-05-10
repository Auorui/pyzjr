import torch.nn as nn

class ConvNormAct(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode='reflect',
            apply_act=True,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.ReLU(inplace=True),
            **kwargs,
    ):
        super(ConvNormAct, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) / 2
        self.apply_act = apply_act
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, **kwargs)
        self.bn = norm_layer(out_channels)
        self.act = act_layer

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.apply_act:
            x = self.act(x)
        return x


class ConvNorm(ConvNormAct):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 norm_layer=nn.BatchNorm2d,
                 **kwargs,):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         apply_act=False,
                         norm_layer=norm_layer,
                         act_layer=nn.ReLU(inplace=True),  # ignore, actually not working
                         **kwargs,)

class NormAct(nn.Module):
    def __init__(self,
                 out_channels,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU(inplace=True),
                 ):
        super(NormAct, self).__init__()
        self.norm_layer = norm_layer(out_channels)
        self.act_layer = act_layer

    def forward(self, x):
        x = self.norm_layer(x)
        x = self.act_layer(x)
        return x

class ConvAct(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
            act_layer=nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)
        self.activation = act_layer

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

def conv3x3(in_channels, out_channels, stride=1, groups=1, padding=1, dilation=1, bias=False):
    """3x3 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation)

def conv1x1(in_channels, out_channels, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)

ConvBnReLU = ConvBNReLU = ConvNormAct
ConvBn = ConvBN = ConvNorm
BnReLU = BNReLU = NormAct
ConvReLU = ConvAct