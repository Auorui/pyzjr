"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
ghostnetv1: <https://arxiv.org/pdf/1911.11907.pdf>
ghostnetv2: <https://arxiv.org/pdf/2211.12905v1.pdf>
g_ghostnet: <https://arxiv.org/pdf/2201.03297.pdf>
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyzjr.nn.models._utils import _make_divisible

__all__ = ["GhostNet", "GGhostRegNet", "ghostnetv1", "ghostnetv2", "g_ghost_regnetx_002", "g_ghost_regnetx_004",
           "g_ghost_regnetx_006", "g_ghost_regnetx_008", "g_ghost_regnetx_016", "g_ghost_regnetx_032",
           "g_ghost_regnetx_040", "g_ghost_regnetx_064", "g_ghost_regnetx_080", "g_ghost_regnetx_120",
           "g_ghost_regnetx_160", "g_ghost_regnetx_320"]

class SeModule(nn.Module):
    def __init__(self, input_channels, se_ratio=0.25, divisor=4):
        super(SeModule, self).__init__()
        reduced_chs = _make_divisible(input_channels * se_ratio, divisor)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, reduced_chs, kernel_size=1, bias=True),
            nn.BatchNorm2d(reduced_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_chs, input_channels, kernel_size=1, bias=True),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        x = x * self.se(x)
        return x

class GhostModulev1(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModulev1, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)   # m = n / s
        new_channels = init_channels*(ratio-1)   # m * (s - 1) = n / s * (s - 1)

        # 利用1x1卷积对输入进来的特征图进行通道的浓缩, 实现跨通道的特征提取
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        # 使用逐层卷积, 获得额外的特征图
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, stride=1, padding=dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneckv1(nn.Module):
    """ Ghost bottleneck w/ optional SE"""
    def __init__(self, in_chs, mid_chs, out_chs, dw_size=3, stride=1, se_ratio=0.):
        super(GhostBottleneckv1, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModulev1(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_size, stride=stride,
                                     padding=(dw_size-1)//2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SeModule(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModulev1(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_size, stride=stride,
                          padding=(dw_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        # 1st ghost bottleneck
        x = self.ghost1(x)
        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)
        # 2nd ghost bottleneck
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x


class GhostModulev2(nn.Module):
    """添加了DFC注意力机制,即short_conv部分,原文使用的是max_pool,这里使用的是avg_pool"""
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, mode=None):
        super(GhostModulev2, self).__init__()
        self.mode = mode
        self.gate_fn = nn.Sigmoid()

        if self.mode in ['original']:
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
        elif self.mode in ['attn']:
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
        if self.mode in ['original']:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, : self.oup, :, :]
        elif self.mode in ['attn']:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, :self.oup, :, :] * F.interpolate(self.gate_fn(res), size=(out.shape[-2], out.shape[-1]), mode='nearest')

class GhostBottleneckv2(nn.Module):
    """与原文有所出入,在消融实验中每层都添加了DFC,这里前两层还是用的原本v1版本"""
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, se_ratio=0., layer_id=None):
        super(GhostBottleneckv2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        if layer_id <= 1:
            self.ghost1 = GhostModulev2(in_chs, mid_chs, relu=True, mode='original')
        else:
            self.ghost1 = GhostModulev2(in_chs, mid_chs, relu=True, mode='attn')

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size-1)//2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SeModule(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        self.ghost2 = GhostModulev2(mid_chs, out_chs, relu=False,mode='original')

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x

class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0, drop_rate=0.2, block=None):
        super(GhostNet, self).__init__()
        self.cfgs = cfgs
        self.drop_rate = drop_rate

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        # 224 x 224 x 3 --> 112 x 112 x 16
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        layer_id = 0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                if block == GhostBottleneckv2:
                    layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                        se_ratio=se_ratio, layer_id=layer_id))
                elif block == GhostBottleneckv1:
                    layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                        se_ratio=se_ratio))
                input_channel = output_channel
                layer_id += 1
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        ))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding:捕捉局部特征和空间相关性，学习更复杂的特征和抽象表示"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution:实现降维或升维，调整通道数和执行通道间的线性变换"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, group_width=1, dilation=1):
        super(Bottleneck, self).__init__()
        width = planes * self.expansion
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, width // min(width, group_width), dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Stage(nn.Module):
    def __init__(self, block, inplanes, planes, group_width, blocks, stride=1, dilate=False, cheap_ratio=0.5):
        super(Stage, self).__init__()
        downsample = None
        self.dilation = 1
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

        self.base = block(inplanes, planes, stride, downsample, group_width,
                          previous_dilation)
        self.end = block(planes, planes, group_width=group_width,
                         dilation=self.dilation,)

        group_width = int(group_width * 0.75)
        raw_planes = int(planes * (1 - cheap_ratio) / group_width) * group_width
        cheap_planes = planes - raw_planes
        self.cheap_planes = cheap_planes
        self.raw_planes = raw_planes

        self.merge = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes + raw_planes * (blocks - 2), cheap_planes,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(cheap_planes, cheap_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(cheap_planes, cheap_planes,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
        )
        self.cheap_relu = nn.ReLU(inplace=True)
        downsample = nn.Sequential(
            LambdaLayer(lambda x: x[:, :raw_planes])
        )

        layers = []
        layers.append(block(raw_planes, raw_planes, 1, downsample, group_width,
                            self.dilation))
        inplanes = raw_planes
        for _ in range(2, blocks - 1):
            layers.append(block(inplanes, raw_planes, group_width=group_width,
                                dilation=self.dilation,))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x0 = self.base(input)

        m_list = [x0]
        e = x0[:, :self.raw_planes]
        for layer in self.layers:
            e = layer(e)
            m_list.append(e)
        m = torch.cat(m_list, 1)
        m = self.merge(m)

        c = x0[:, self.raw_planes:]
        c = self.cheap_relu(self.cheap(c) + m)

        x = torch.cat((e, c), 1)
        x = self.end(x)
        return x


class GGhostRegNet(nn.Module):
    def __init__(self, block, layers, widths, num_classes=1000,
                 group_width=1):
        super(GGhostRegNet, self).__init__()
        self.inplanes = 32
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False, False]
        self.group_width = group_width
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, widths[0], layers[0], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        self.inplanes = widths[0]
        if layers[1] > 2:
            self.layer2 = Stage(block, self.inplanes, widths[1], group_width, layers[1], stride=2,
                                dilate=replace_stride_with_dilation[1], cheap_ratio=0.5)
        else:
            self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[1])

        self.inplanes = widths[1]
        self.layer3 = Stage(block, self.inplanes, widths[2], group_width, layers[2], stride=2,
                            dilate=replace_stride_with_dilation[2], cheap_ratio=0.5)

        self.inplanes = widths[2]
        if layers[3] > 2:
            self.layer4 = Stage(block, self.inplanes, widths[3], group_width, layers[3], stride=2,
                                dilate=replace_stride_with_dilation[3], cheap_ratio=0.5)
        else:
            self.layer4 = self._make_layer(block, widths[3], layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(widths[-1] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.group_width,
                            previous_dilation))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group_width=self.group_width,
                                dilation=self.dilation,))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x



def g_ghost_regnetx_002(**kwargs):
    return GGhostRegNet(Bottleneck, [1, 1, 4, 7], [24, 56, 152, 368], group_width=8, **kwargs)


def g_ghost_regnetx_004(**kwargs):
    return GGhostRegNet(Bottleneck, [1, 2, 7, 12], [32, 64, 160, 384], group_width=16, **kwargs)


def g_ghost_regnetx_006(**kwargs):
    return GGhostRegNet(Bottleneck, [1, 3, 5, 7], [48, 96, 240, 528], group_width=24, **kwargs)


def g_ghost_regnetx_008(**kwargs):
    return GGhostRegNet(Bottleneck, [1, 3, 7, 5], [64, 128, 288, 672], group_width=16, **kwargs)


def g_ghost_regnetx_016(**kwargs):
    return GGhostRegNet(Bottleneck, [2, 4, 10, 2], [72, 168, 408, 912], group_width=24, **kwargs)


def g_ghost_regnetx_032(**kwargs):
    return GGhostRegNet(Bottleneck, [2, 6, 15, 2], [96, 192, 432, 1008], group_width=48, **kwargs)


def g_ghost_regnetx_040(**kwargs):
    return GGhostRegNet(Bottleneck, [2, 5, 14, 2], [80, 240, 560, 1360], group_width=40, **kwargs)


def g_ghost_regnetx_064(**kwargs):
    return GGhostRegNet(Bottleneck, [2, 4, 10, 1], [168, 392, 784, 1624], group_width=56, **kwargs)


def g_ghost_regnetx_080(**kwargs):
    return GGhostRegNet(Bottleneck, [2, 5, 15, 1], [80, 240, 720, 1920], group_width=120, **kwargs)


def g_ghost_regnetx_120(**kwargs):
    return GGhostRegNet(Bottleneck, [2, 5, 11, 1], [224, 448, 896, 2240], group_width=112, **kwargs)


def g_ghost_regnetx_160(**kwargs):
    return GGhostRegNet(Bottleneck, [2, 6, 13, 1], [256, 512, 896, 2048], group_width=128, **kwargs)


def g_ghost_regnetx_320(**kwargs):
    return GGhostRegNet(Bottleneck, [2, 7, 13, 1], [336, 672, 1344, 2520], group_width=168, **kwargs)


def ghostnetv1(num_classes):
    """
    * k: 卷积核大小,表示跨特征点提取特征的能力
    * t: 第一个ghost模块的通道数大小
    * c: 瓶颈结构最终输出通道数的大小
    * SE: 是否使用注意力模块
    * s: 步长大小,如果为2,则会进行高和宽的压缩
    """
    cfgs = [
        # k,  t,   c,  SE,  s
        # stage1
        [[3,  16,  16,  0,  1]],      # 112 x 112 x 16 --> 112 x 112 x 16
        # stage2
        [[3,  48,  24,  0,  2]],      # 112 x 112 x 16 --> 56 x 56 x 24
        [[3,  72,  24,  0,  1]],
        # stage3
        [[5,  72,  40, 0.25, 2]],     # 56 x 56 x 24 --> 28 x 28 x 40
        [[5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80,  0,  2]],      # 28 x 28 x 40 --> 14 x 14 x 80
        [[3, 200,  80,  0,  1],
         [3, 184,  80,  0,  1],
         [3, 184,  80,  0,  1],
         [3, 480, 112, 0.25, 1],      # 14 x 14 x 80 --> 14 x 14 x 112
         [3, 672, 112, 0.25, 1]],
        # stage5
        [[5, 672, 160, 0.25, 2]],     # 14 x 14 x 112 --> 7 x 7 x 160
        [[5, 960, 160,  0,  1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160,  0,  1],
         [5, 960, 160, 0.25, 1]]
    ]
    return GhostNet(cfgs, num_classes=num_classes, block=GhostBottleneckv1)

def ghostnetv2(num_classes):
    """
    * k: 卷积核大小,表示跨特征点提取特征的能力
    * t: 第一个ghost模块的通道数大小
    * c: 瓶颈结构最终输出通道数的大小
    * SE: 是否使用注意力模块
    * s: 步长大小,如果为2,则会进行高和宽的压缩
    """
    cfgs = [
        # k,   t,  c,  SE,  s
        # stage1
        [[3,  16,  16,  0,  1]],      # 112 x 112 x 16 --> 112 x 112 x 16
        # stage2
        [[3,  48,  24,  0,  2]],      # 112 x 112 x 16 --> 56 x 56 x 24
        [[3,  72,  24,  0,  1]],
        # stage3
        [[5,  72,  40, 0.25, 2]],     # 56 x 56 x 24 --> 28 x 28 x 40
        [[5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80,  0,  2]],      # 28 x 28 x 40 --> 14 x 14 x 80
        [[3, 200,  80,  0,  1],
         [3, 184,  80,  0,  1],
         [3, 184,  80,  0,  1],
         [3, 480, 112, 0.25, 1],      # 14 x 14 x 80 --> 14 x 14 x 112
         [3, 672, 112, 0.25, 1]],
        # stage5
        [[5, 672, 160, 0.25, 2]],     # 14 x 14 x 112 --> 7 x 7 x 160
        [[5, 960, 160,  0,  1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160,  0,  1],
         [5, 960, 160, 0.25, 1]]
    ]
    return GhostNet(cfgs, num_classes=num_classes, block=GhostBottleneckv2)



if __name__=="__main__":
    import torchsummary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = ghostnetv2(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    torchsummary.summary(net, input_size=(3, 224, 224))
    # v1 Total params: 3,908,608
    # v2 Total params: 4,883,008