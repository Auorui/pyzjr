"""
Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ["DenseNet", "densenet121", "densenet161", "densenet169", "densenet201"]

class Transition(nn.Sequential):
    """
    减少通道数, 特征图尺寸减半
    Densenet Transition Layer:
        1 × 1 conv
        2 × 2 average pool, stride 2
    """
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

class DenseBlock(nn.Module):
    def __init__(
            self,
            num_layers,
            num_input_features,
            bn_size,
            growth_rate,
            drop_rate=0.0
    ):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        num_features = num_input_features
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

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, num_init_features=64, block_config=None, num_classes = 1000,
                 bn_size=4, drop_rate=0.):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:   # 层与层之间添加过渡层
                trans = Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def densenet121(num_classes):
    """Densenet-121 model"""
    return DenseNet(32, 64, (6, 12, 24, 16),num_classes=num_classes)

def densenet161(num_classes):
    """Densenet-161 model"""
    return DenseNet(48, 96, (6, 12, 36, 24),  num_classes=num_classes)

def densenet169(num_classes):
    """Densenet-169 model"""
    return DenseNet(32, 64, (6, 12, 32, 32),   num_classes=num_classes)

def densenet201(num_classes):
    """Densenet-201 model"""
    return DenseNet(32, 64, (6, 12, 48, 32), num_classes=num_classes)


if __name__=="__main__":
    from pyzjr.nn.tools import summary_2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = densenet121(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    summary_2(net, torch.ones((1, 3, 224, 224)).to(device))
    # Params:    6,957.96K
