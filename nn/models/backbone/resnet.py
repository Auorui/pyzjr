"""
Deep Residual Learning for Image Recognition
    <https://arxiv.org/pdf/1512.03385.pdf>
resnet18, resnet34, resnet50, resnet101, resnet152
(Optional addition of SE module)

Blog records: https://blog.csdn.net/m0_62919535/article/details/132384303
"""
import torch
import torch.nn as nn
from pyzjr.nn.models.atten import SEAttention
from pyzjr.nn.models.bricks.conv_norm_act import ConvBnReLU, ConvBn

__all__ = ["ResNet", 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding:捕捉局部特征和空间相关性，学习更复杂的特征和抽象表示"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution:实现降维或升维，调整通道数和执行通道间的线性变换"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16, attn=False):
        super(BasicBlock, self).__init__()
        self.convbnrelu1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.convbn1 = ConvBn(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.se = SEAttention(out_channels, reduction)
        self.attn = attn
        self.conv_down = nn.Sequential(
            conv1x1(in_channels, out_channels * self.expansion, self.stride),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

    def forward(self, x):
        residual = x
        out = self.convbnrelu1(x)
        out = self.convbn1(out)
        if self.attn:
            out = self.se(out)

        if self.downsample:
            residual = self.conv_down(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16, attn=False):
        super(Bottleneck, self).__init__()
        groups = 1
        base_width = 64
        dilation = 1

        width = int(out_channels * (base_width / 64.)) * groups   # wide = out_channels
        self.conv1 = conv1x1(in_channels, width)       # 降维通道数
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)   # 升维通道数
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.se = SEAttention(out_channels * self.expansion, reduction)
        self.attn = attn
        self.conv_down = nn.Sequential(
            conv1x1(in_channels, out_channels * self.expansion, self.stride),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

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

        if self.attn:
            out = self.se(out)

        if self.downsample:
            residual = self.conv_down(x)

        out += residual
        out = self.relu(out)

        return out

class SEBasicBlock(BasicBlock):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__(in_channels, out_channels, stride, downsample, reduction, attn=True)

class SEBottleneck(Bottleneck):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__(in_channels, out_channels, stride, downsample, reduction, attn=True)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = False
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = True

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):    # 添加几个残差块, 跟resnet的结构有关
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # torch.Size([2, 64, 56, 56])
        x = self.layer1(x)     # torch.Size([2, 256, 56, 56])
        x = self.layer2(x)     # torch.Size([2, 512, 56, 56])
        x = self.layer3(x)     # torch.Size([2, 1024, 28, 28])
        x = self.layer4(x)     # torch.Size([2, 2048, 14, 14])

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18(num_classes, use_se=False, **kwargs):
    block = SEBasicBlock if use_se else BasicBlock
    return ResNet(block, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

def resnet34(num_classes, use_se=False, **kwargs):
    block = SEBasicBlock if use_se else BasicBlock
    return ResNet(block, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def resnet50(num_classes, use_se=False, **kwargs):
    block = SEBottleneck if use_se else Bottleneck
    return ResNet(block, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def resnet101(num_classes, use_se=False, **kwargs):
    block = SEBottleneck if use_se else Bottleneck
    return ResNet(block, [3, 4, 23, 3], num_classes=num_classes, **kwargs)

def resnet152(num_classes, use_se=False, **kwargs):
    block = SEBottleneck if use_se else Bottleneck
    return ResNet(block, [3, 8, 36, 3], num_classes=num_classes, **kwargs)

if __name__=="__main__":
    import torchsummary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = resnet34(num_classes=4, use_se=True)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    torchsummary.summary(net, input_size=(3, 224, 224))
    # 50 Total params: 23,516,228