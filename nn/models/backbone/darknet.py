"""
Copyright (c) 2024, Auorui.
All rights reserved.

Darknet-19: https://arxiv.org/pdf/1612.08242.pdf                     Yolov2
Darknet-53: https://pjreddie.com/media/files/papers/YOLOv3.pdf       Yolov3
Blog records: https://blog.csdn.net/m0_62919535/article/details/132639078
"""
import torch
import math
from torch import nn
from collections import OrderedDict

__all__ = ["darknet19", "darknet53"]

class ConvBNLR(nn.Sequential):
    """
    ConvBNLR -> Conv+BN+LeakyReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, negative_slope=0.1, **kwargs):
        super(ConvBNLR, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope)
        )

class DarkResidualBlock(nn.Module):
    # Residual block
    def __init__(self, inplanes, planes):
        super(DarkResidualBlock, self).__init__()
        # block1 降通道数，block2 再将通道数升回去，如 64->32->64
        self.conv = nn.Sequential(
            ConvBNLR(inplanes, planes[0], kernel_size=1, padding=0),
            ConvBNLR(planes[0], planes[1])
        )

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        return out


class Darknet53(nn.Module):
    def __init__(self, num_classes, layers):
        super(Darknet53, self).__init__()
        self.inplanes = 32
        self.layer0 = ConvBNLR(3, self.inplanes)                    # (3, 224, 224)   -> (32, 224, 224)
        self.layer1 = self._make_layer([32, 64], layers[0])         # (32, 224, 224)  -> (64, 112, 112)
        self.layer2 = self._make_layer([64, 128], layers[1])        # (64, 112, 112)  -> (128, 56, 56)
        self.layer3 = self._make_layer([128, 256], layers[2])       # (128, 56, 56) -> (256, 28, 28)
        self.layer4 = self._make_layer([256, 512], layers[3])       # (256, 28, 28)   -> (512, 14, 14)
        self.layer5 = self._make_layer([512, 1024], layers[4])      # (512, 14, 14)   -> (1024, 7, 7)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.layer5(x)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)

        return out

    def _make_layer(self, planes, blocks):
        """
        在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
        然后进行残差结构的堆叠
        """
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("down_sampling", ConvBNLR(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), DarkResidualBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

class Darknet19(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Darknet19, self).__init__()
        self.features = self.make_layers(cfg)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)  # Assuming the last value before 'M' is the output channels

    def make_layers(self, cfg):
        """参考的VGG的复用结构"""
        layers = []
        in_channels = 3
        for i in cfg:
            if i == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels, kernel_size = i
                padding = 1 if kernel_size > 1 else 0
                layers.append(ConvBNLR(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Model structure reference from https://arxiv.org/pdf/1612.08242.pdf Table 6 —— Darknet19
#                                https://pjreddie.com/media/files/papers/YOLOv3.pdf Table 1 —— Darknet53
cfg = {"A": [(32, 3), 'M', (64, 3), 'M', (128, 3), (64, 1), (128, 3), 'M', (256, 3), (128, 1),
             (256, 3), 'M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3), 'M', (1024, 3),
             (512, 1), (1024, 3), (512, 1), (1024, 3)],

       "B": [1, 2, 8, 8, 4]}

def darknet19(num_classes):
    return Darknet19(num_classes, cfg["A"])

def darknet53(num_classes):
    return Darknet53(num_classes, cfg["B"])

if __name__=="__main__":
    import torchsummary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = darknet19(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    torchsummary.summary(net, input_size=(3, 224, 224))
    # Total params: 19,821,476  --19
    # Total params: 40,589,028  --53