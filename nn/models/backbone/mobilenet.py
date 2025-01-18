"""
Copyright (c) 2024, Auorui.
All rights reserved.

v1: <https://arxiv.org/pdf/1704.04861.pdf>
v2: <https://arxiv.org/pdf/1801.04381.pdf>
v3: <https://arxiv.org/abs/1905.02244.pdf>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from pyzjr.nn.models._utils import _make_divisible

__all__ = ["MobileNetV1", "MobileNetV2", "MobileNetV3", "MobileNetV3_Small", "MobileNetV3_Large"]

class DepthSepConv(nn.Module):
    """
    深度可分卷积: DW卷积 + PW卷积
    dw卷积, 当分组个数等于输入通道数时, 输出矩阵的通道输也变成了输入通道数
    pw卷积, 使用了1x1的卷积核与普通的卷积一样
    """
    def __init__(self, in_channels, out_channels, stride):
        super(DepthSepConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.batch_norm1(x)
        x = self.relu6(x)
        x = self.pointwise(x)
        x = self.batch_norm2(x)
        x = self.relu6(x)

        return x

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, drop_rate=0.2):
        super(MobileNetV1, self).__init__()
        # torch.Size([1, 3, 224, 224])
        self.conv_bn = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )                                # torch.Size([1, 32, 112, 112])
        self.dwmodule = nn.Sequential(
            # 参考MobileNet_V1 https://arxiv.org/pdf/1704.04861.pdf Table 1
            DepthSepConv(32, 64, 1),            # torch.Size([1, 64, 112, 112])
            DepthSepConv(64, 128, 2),           # torch.Size([1, 128, 56, 56])
            DepthSepConv(128, 128, 1),          # torch.Size([1, 128, 56, 56])
            DepthSepConv(128, 256, 2),          # torch.Size([1, 256, 28, 28])
            DepthSepConv(256, 256, 1),          # torch.Size([1, 256, 28, 28])
            DepthSepConv(256, 512, 2),          # torch.Size([1, 512, 14, 14])
            # 5 x DepthSepConv(512, 512, 1),
            DepthSepConv(512, 512, 1),          # torch.Size([1, 512, 14, 14])
            DepthSepConv(512, 512, 1),
            DepthSepConv(512, 512, 1),
            DepthSepConv(512, 512, 1),
            DepthSepConv(512, 512, 1),
            DepthSepConv(512, 1024, 2),         # torch.Size([1, 1024, 7, 7])
            DepthSepConv(1024, 1024, 1),
            nn.AvgPool2d(7, stride=1),
        )
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)
        self.dropout = nn.Dropout(p=drop_rate)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_bn(x)
        x = self.dwmodule(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(self.dropout(x))
        return x

class ConvBNActivation(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1,
                 norm_layer=None, activation_layer=None, dilation=1,):
        super(ConvBNActivation, self).__init__()
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        self.convbnact=nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes

    def forward(self, x):
        return self.convbnact(x)

class InvertedResidualv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride, expand_ratio):
        super(InvertedResidualv2, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(in_planes * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_planes == out_planes

        layers = []
        if expand_ratio != 1:
            # pw 利用1x1卷积进行通道数的上升
            layers.append(ConvBNActivation(in_planes, hidden_dim, kernel_size=1))

        layers.extend([
            # dw 进行3x3的逐层卷积，进行跨特征点的特征提取
            ConvBNActivation(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim),
            # pw-linear 利用1x1卷积进行通道数的下降
            nn.Conv2d(hidden_dim, out_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_planes),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = out_planes

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, drop_rate=0.2, width_mult=1.0, round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            drop_rate (float): Dropout layer drop rate
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        input_channel = 32
        last_channel = 1280
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # t表示是否进行1*1卷积上升的过程 c表示output_channel大小 n表示小列表倒残差次数 s是步长,表示是否对高和宽进行压缩
        # building first layer
        features = [ConvBNActivation(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidualv2(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNActivation(input_channel, last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = F.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

class SeModule(nn.Module):
    def __init__(self, input_channels, reduction=4):
        super(SeModule, self).__init__()
        expand_size = _make_divisible(input_channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, input_channels, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MobileNetV3(nn.Module):
    """
    MobileNet V3 main class

    Args:
        num_classes: Number of classes
        mode: "large" or "small"
    """
    def __init__(self, num_classes=1000, mode=None, drop_rate=0.2):
        super().__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        layers = []
        inverted_residual_setting, last_channel = _mobilenetv3_cfg[mode]
        # building first layer
        firstconv_output_channels = 16
        layers.append(ConvBNActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        layers.append(inverted_residual_setting)
        # building last several layers
        lastconv_input_channels = 96 if mode == "small" else 160
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvBNActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                       norm_layer=norm_layer, activation_layer=nn.Hardswish))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=drop_rate, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

class InvertedResidualv3(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, input_channels, expanded_channels, out_channels, activation, use_se, stride):
        super(InvertedResidualv3, self).__init__()
        self.stride = stride
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        self.use_res_connect = stride == 1 and input_channels == out_channels
        activation_layer = nn.ReLU if activation == "RE" else nn.Hardswish
        layers = []
        if expanded_channels != input_channels:
            layers.append(ConvBNActivation(input_channels, expanded_channels, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        layers.append(ConvBNActivation(expanded_channels, expanded_channels, kernel_size=kernel_size,
                                       stride=stride, groups=expanded_channels,
                                       norm_layer=norm_layer, activation_layer=activation_layer))
        if use_se:
            layers.append(SeModule(expanded_channels))

        layers.append(ConvBNActivation(expanded_channels, out_channels, kernel_size=1, norm_layer=norm_layer,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result

_mobilenetv3_cfg = {
    "large": [nn.Sequential(
        # kernel, in_chs, exp_chs, out_chs, act, use_se, stride
        InvertedResidualv3(3, 16, 16, 16, "RE", False, 1),
        InvertedResidualv3(3, 16, 64, 24, "RE", False, 2),
        InvertedResidualv3(3, 24, 72, 24, "RE", False, 1),
        InvertedResidualv3(5, 24, 72, 40, "RE", True, 2),
        InvertedResidualv3(5, 40, 120, 40, "RE", True, 1),
        InvertedResidualv3(5, 40, 120, 40, "RE", True, 1),
        InvertedResidualv3(3, 40, 240, 80, "HS", False, 2),
        InvertedResidualv3(3, 80, 200, 80, "HS", False, 1),
        InvertedResidualv3(3, 80, 184, 80, "HS", False, 1),
        InvertedResidualv3(3, 80, 184, 80, "HS", False, 1),
        InvertedResidualv3(3, 80, 480, 112, "HS", True, 1),
        InvertedResidualv3(3, 112, 672, 112, "HS", True, 1),
        InvertedResidualv3(5, 112, 672, 160, "HS", True, 1),
        InvertedResidualv3(5, 160, 672, 160, "HS", True, 2),
        InvertedResidualv3(5, 160, 960, 160, "HS", True, 1),
    ),
        _make_divisible(1280, 8)],
    "small": [nn.Sequential(
        # kernel, in_chs, exp_chs, out_chs, act, use_se, stride
        InvertedResidualv3(3, 16, 16, 16, "RE", True, 2),
        InvertedResidualv3(3, 16, 72, 24, "RE", False, 2),
        InvertedResidualv3(3, 24, 88, 24, "RE", False, 1),
        InvertedResidualv3(5, 24, 96, 40, "HS", True, 2),
        InvertedResidualv3(5, 40, 240, 40, "HS", True, 1),
        InvertedResidualv3(5, 40, 240, 40, "HS", True, 1),
        InvertedResidualv3(5, 40, 120, 48, "HS", True, 1),
        InvertedResidualv3(5, 48, 144, 48, "HS", True, 1),
        InvertedResidualv3(5, 48, 288, 96, "HS", True, 2),
        InvertedResidualv3(5, 96, 576, 96, "HS", True, 1),
        InvertedResidualv3(5, 96, 576, 96, "HS", True, 1),
    ),
        _make_divisible(1024, 8)],
}

def MobileNetV3_Large(num_classes):
    """Large version of mobilenet_v3"""
    return MobileNetV3(num_classes=num_classes, mode="large")

def MobileNetV3_Small(num_classes):
    """small version of mobilenet_v3"""
    return MobileNetV3(num_classes=num_classes, mode="small")

if __name__=="__main__":
    import torchsummary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = MobileNetV2(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    torchsummary.summary(net, input_size=(3, 224, 224))
    # v1 Total params: 3,221,988
    # v2 Total params: 2,230,500
    # v3_small Total params: 1,520,252
    # v3_large Total params: 3,868,460