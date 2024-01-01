import math
import torch.nn as nn
from collections import OrderedDict

__all__ = ["darknet53"]

class CBLR(nn.Sequential):
    """
    CBLR -> Conv+BN+LeakyReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, negative_slope=0.1):
        super(CBLR, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope)
        )

class BasicBlock_with_darknet(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock_with_darknet, self).__init__()
        # block1 降通道数，block2 再将通道数升回去，如 64->32->64
        self.conv = nn.Sequential(
            CBLR(inplanes, planes[0], kernel_size=1, padding=0),
            CBLR(planes[0], planes[1])
        )

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        self.layer0 = CBLR(3, self.inplanes)                        # (3, 416, 416)   -> (32, 416, 416)
        self.layer1 = self._make_layer([32, 64], layers[0])         # (32, 416, 416)  -> (64, 208, 208)
        self.layer2 = self._make_layer([64, 128], layers[1])        # (64, 208, 208)  -> (128, 104, 104)
        self.layer3 = self._make_layer([128, 256], layers[2])       # (128, 104, 104) -> (256, 52, 52)
        self.layer4 = self._make_layer([256, 512], layers[3])       # (256, 52, 52)   -> (512, 26, 26)
        self.layer5 = self._make_layer([512, 1024], layers[4])      # (512, 26, 26)   -> (1024, 13, 13)

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        """
        在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
        然后进行残差结构的堆叠
        """
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 加入残差结构
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock_with_darknet(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5

def darknet53():
    model = DarkNet([1, 2, 8, 8, 4])
    return model