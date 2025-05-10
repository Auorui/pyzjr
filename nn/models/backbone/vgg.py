"""
论文原址： <https://arxiv.org/pdf/1409.1556v6.pdf>
VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION
VGG16/VGG19 bn
Time: 2023-10-19
"""
import torch
import torch.nn as nn

__all__ = ["VGG", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]

cfg = {
    'vgg11': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg):
    layers = nn.ModuleList()
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg11_bn(num_classes):
    model = VGG(make_layers(cfg['vgg11']), num_classes=num_classes)
    return model

def vgg13_bn(num_classes):
    model = VGG(make_layers(cfg['vgg13']), num_classes=num_classes)
    return model

def vgg16_bn(num_classes):
    model = VGG(make_layers(cfg['vgg16']), num_classes=num_classes)
    return model

def vgg19_bn(num_classes):
    model = VGG(make_layers(cfg['vgg19']), num_classes=num_classes)
    return model

if __name__=='__main__':
    import torchsummary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = vgg11_bn(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    torchsummary.summary(net, input_size=(3, 224, 224))
    # Total params: 128,788,228
    # Total params: 128,973,124        11
    # Total params: 134,285,380        16
    # Total params: 139,597,636        19