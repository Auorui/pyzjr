"""
Original paper addresshttps: https://arxiv.org/pdf/1709.01507.pdf
Code originates from: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
Blog records: https://blog.csdn.net/m0_62919535/article/details/135761713
Time: 2024-01-23
"""
import torch
from torch import nn

class SEAttention(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SqueezeExcitation(nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.
    """
    def __init__(
            self,
            input_channels,
            squeeze_channels,
            activation=torch.nn.ReLU,
            scale_activation=torch.nn.Sigmoid):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return scale * input


class EffectiveSEAttention(nn.Module):
    """ 'Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    code come from: https://github.com/youngwanLEE/CenterMask/blob/master/maskrcnn_benchmark/modeling/backbone/vovnet.py
    """
    def __init__(self, channel):
        super(EffectiveSEAttention, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1,
                            padding=0)
        self.hsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        input = x
        x = x.mean((2, 3), keepdim=True)   # x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x


if __name__ == '__main__':
    input = torch.randn(1, 64, 64, 64)
    se = EffectiveSEAttention(channel=64)
    output = se(input)
    print(output.shape)
