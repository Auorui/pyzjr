"""
VGG11/13/16/19 in Pytorch.
Very Deep Convolutional Networks for Large-Scale Image Recognition.
    -> https://arxiv.org/abs/1409.1556v6
"""
import torch
import torch.nn as nn

__all__ = ["vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn", "VGG"]

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True, dropout=0.5):
        super(VGG,self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn(num_class):
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes=num_class)

def vgg13_bn(num_class):
    return VGG(make_layers(cfg['B'], batch_norm=True), num_classes=num_class)

def vgg16_bn(num_class):
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes=num_class)

def vgg19_bn(num_class):
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes=num_class)


if __name__=='__main__':
    import torchsummary
    in_data = torch.ones(2, 3, 224, 224).cuda()
    net = vgg16_bn(num_class=2)
    net = net.cuda()
    out = net(in_data)
    print(out)
    torchsummary.summary(net, input_size=(3, 224, 224))
