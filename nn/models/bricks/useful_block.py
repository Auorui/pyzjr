import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyzjr.nn.models.atten.se_module import SEAttention
from pyzjr.nn.models.bricks.conv_norm_act import ConvBNReLU, ConvBN, conv1x1


class ResBasicBlock(nn.Module):
    """The infrastructure of ResNet"""
    expansion = 1   # The output channel of a basic block is usually twice that of the input channel
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            use_se=False,
            reduction=16
    ):
        super(ResBasicBlock, self).__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ConvBN(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.attention = SEAttention(out_channels, reduction)
        self.use_se = use_se
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels * self.expansion, stride),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_se:
            out = self.attention(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResBottleneck(nn.Module):
    """The bottleneck structure of ResNet"""
    expansion = 4
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            use_se=False,
            reduction=16
    ):
        super(ResBottleneck, self).__init__()
        groups = 1
        base_width = 64
        dilation = 1

        width = int(out_channels * (base_width / 64.)) * groups   # wide = out_channels
        self.convbnrelu1 = ConvBNReLU(in_channels, width, kernel_size=1, padding=0)  # 降维通道数
        self.convbnrelu2 = ConvBNReLU(width, width, kernel_size=3, padding=1, stride=stride, dilation=dilation, groups=groups)
        self.convbn3 = ConvBN(width, out_channels * self.expansion, kernel_size=1, padding=0)   # 升维通道数
        self.attention = SEAttention(out_channels * self.expansion, reduction)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.use_se = use_se
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels * self.expansion, stride),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.convbnrelu1(x)
        out = self.convbnrelu2(out)
        out = self.convbn3(out)
        if self.use_se:
            out = self.attention(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class FireModule(nn.Module):
    """SqueezeNet's compression and expansion of channels"""
    def __init__(self, in_channels, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_planes, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.squeeze(x))
        return torch.cat([
            self.relu(self.expand1x1(x)),
            self.relu(self.expand3x3(x))
        ], dim=1)

class DenseBlock(nn.Module):
    """Dense connection module"""
    def __init__(
            self,
            num_layers,
            in_channels,
            bn_size,
            growth_rate,
            drop_rate=0.0
    ):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        num_features = in_channels
        for i in range(num_layers):
            layer = self._make_dense_layer(num_features, growth_rate, bn_size, drop_rate)
            self.layers.append(layer)
            num_features += growth_rate

    def _make_dense_layer(self, num_input_features, growth_rate, bn_size, drop_rate):
        layers = [
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        ]
        if drop_rate > 0:
            layers.append(nn.Dropout(p=drop_rate, inplace=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        for layer in self.layers:
            new_features = layer(out)
            out = torch.cat([out, new_features], 1)
        return out

class MobileInvertedResidual(nn.Module):
    """The Reverse Residual Structure of MobileNetV2"""
    def __init__(
            self,
            in_channels,
            out_channels,
            stride,
            expand_ratio
    ):
        super(MobileInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1, padding=0))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim,
                       padding=1 if stride == 1 else 0, act_layer=nn.ReLU6(inplace=True)),
            ConvBN(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class GhostModule(nn.Module):
    """Cheap operation module for GhostNet"""
    def __init__(
            self,
            in_channels,
            out_channels,
            use_dfc=True,
            kernel_size=1,
            dw_kernel_size=3,
            ratio=2,
            stride=1,
            relu=True,
    ):
        super(GhostModule, self).__init__()
        self.use_dfc = use_dfc
        self.gate_fn = nn.Sigmoid()
        self.oup = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels*(ratio-1)
        self.primary_conv = nn.Sequential(
            ConvBN(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            ConvBN(init_channels, new_channels, dw_kernel_size, 1, dw_kernel_size//2, groups=init_channels, bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.dfc = nn.Sequential(
            # horizontal DFC and vertical DFC
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), stride=1, padding=(0,2), groups=out_channels,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), stride=1, padding=(2,0), groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        if self.use_dfc:
            res = self.dfc(x)
            return out[:, :self.oup, :, :] * F.interpolate(self.gate_fn(res), size=(out.shape[-2], out.shape[-1]), mode='nearest')
        else:
            return out[:, :self.oup, :, :]

class GhostBottleneck(nn.Module):
    """The Cheap Operation Bottleneck Structure of GhostNet"""
    def __init__(
            self,
            in_channels,
            mid_channels,
            out_channels,
            dw_kernel_size=3,
            stride=1,
            layer_id=None
    ):
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        # Point-wise expansion
        if layer_id <= 1:
            self.ghost1 = GhostModule(in_channels, mid_channels, use_dfc=False)
        else:
            self.ghost1 = GhostModule(in_channels, mid_channels, use_dfc=True)
        # Depth-wise convolution
        if self.stride > 1:
            self.conv_bn_dw = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, dw_kernel_size, stride=stride,
                    padding=(dw_kernel_size-1)//2, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels)
            )
        self.ghost2 = GhostModule(mid_channels, out_channels, relu=False, use_dfc=False)

        # shortcut
        if (in_channels == out_channels and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                ConvBN(in_channels, out_channels, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_channels, bias=False),
                ConvBN(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
            )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_bn_dw(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x

if __name__=="__main__":
    input = torch.randn(32, 64, 56, 56)
    # block = GhostBottleneck(in_channels=64, mid_channels=128, out_channels=64, layer_id=0)
    # block = MobileInvertedResidual(in_channels=64, out_channels=128, stride=1, expand_ratio=6)
    # block = DenseBlock(num_layers=4, in_channels=64, bn_size=4, growth_rate=12, drop_rate=0.0)
    # block = FireModule(64, squeeze_planes=16, expand1x1_planes=64, expand3x3_planes=64)
    block = ResBottleneck(in_channels=64, out_channels=64, stride=1, use_se=True)
    output = block(input)
    print(input.shape, output.shape)