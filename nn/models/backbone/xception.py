"""
Copyright (c) 2023, Auorui.
All rights reserved.

Xception: Deep Learning with Depthwise Separable Convolutions
    <https://arxiv.org/pdf/1610.02357.pdf>

Blog records: https://blog.csdn.net/m0_62919535/article/details/137065439
"""
import torch
import torch.nn as nn

__all__ = ["Xception"]

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                               dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                   dilation=1, groups=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise(x)
        return x

class EntryFlow(nn.Module):
    def __init__(self):
        super(EntryFlow, self).__init__()

        self.headconv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.residual_block1 = nn.Sequential(
            SeparableConv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SeparableConv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.residual_block2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SeparableConv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.residual_block3 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv2d(256, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            SeparableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

    def shortcut(self, inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 2, bias=False),
            nn.BatchNorm2d(oup)
            )

    def forward(self, x):
        x = self.headconv(x)
        residual = self.residual_block1(x)
        shortcut_block1 = self.shortcut(64, 128)
        x = residual + shortcut_block1(x)
        residual = self.residual_block2(x)
        shortcut_block2 = self.shortcut(128, 256)
        x = residual + shortcut_block2(x)
        residual = self.residual_block3(x)
        shortcut_block3 = self.shortcut(256, 728)
        x = residual + shortcut_block3(x)

        return x

class MiddleFlow(nn.Module):
    def __init__(self):
        super(MiddleFlow, self).__init__()

        self.shortcut = nn.Sequential()
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),

            nn.ReLU(inplace=True),
            SeparableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),

            nn.ReLU(inplace=True),
            SeparableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        residual = self.conv1(x)
        input = self.shortcut(x)
        return input + residual

class ExitFlow(nn.Module):
    def __init__(self):
        super(ExitFlow, self).__init__()

        self.residual_with_exit = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            SeparableConv2d(728, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.endconv = nn.Sequential(
            SeparableConv2d(1024, 1536, 3, 1, 1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),

            SeparableConv2d(1536, 2048, 3, 1, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def shortcut(self, inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 2, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        residual = self.residual_with_exit(x)
        shortcut_block = self.shortcut(728, 1024)
        output = residual + shortcut_block(x)
        return self.endconv(output)


class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes

        self.entry_flow = EntryFlow()
        self.middle_flow = MiddleFlow()
        self.exit_flow = ExitFlow()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.entry_flow(x)
        for i in range(8):
            # Repeated 8 times
            x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out

if __name__=='__main__':
    import torchsummary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = Xception(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    torchsummary.summary(net, input_size=(3, 224, 224))
    # Xception Total params: 19,838,076