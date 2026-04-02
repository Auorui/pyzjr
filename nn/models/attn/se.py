"""
Original paper addresshttps: https://arxiv.org/pdf/1709.01507.pdf
Code originates from: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
Blog records: https://blog.csdn.net/m0_62919535/article/details/135761713
Time: 2024-01-23
"""
import torch
import torch.nn as nn

class SEAttention(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EffectiveSEAttention(nn.Module):
    """ 'Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    code come from: https://github.com/youngwanLEE/CenterMask/blob/master/maskrcnn_benchmark/modeling/backbone/vovnet.py
    """
    def __init__(self, dim):
        super(EffectiveSEAttention, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(dim, dim, kernel_size=1,
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
    se =SEAttention(dim=64)
    output = se(input)
    print(output.shape)
    se2 = EffectiveSEAttention(dim=64)
    output = se2(input)
    print(output.shape)