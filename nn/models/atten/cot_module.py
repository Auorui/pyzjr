"""
可用于替代普通的3x3卷积
Original paper addresshttps: <https://arxiv.org/pdf/2107.12292.pdf>
The code comes from the official implementation:
https://github.com/JDAI-CV/CoTNet
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

class CoTAttention(nn.Module):
    def __init__(self, in_planes, kernel_size=3, padding=None):
        super().__init__()
        self.dim = in_planes
        self.kernel_size = kernel_size
        if padding is None:
            padding = kernel_size // 2

        self.key_embed = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=kernel_size, padding=padding,
                      groups=4, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 1, bias=False),
            nn.BatchNorm2d(self.dim)
        )

        factor = 4
        # 类似于 SEAttention 结构
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * self.dim, 2 * self.dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * self.dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * self.dim // factor, kernel_size * kernel_size * self.dim, 1)
        )


    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)    # bs, c, h, w
        v = self.value_embed(x).view(bs, c, -1)    # bs, c, h * w

        y = torch.cat([k1, x], dim=1)      # bs, 2c, h * w
        att = self.attention_embed(y)        # bs, c * k * k, h, w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)        # bs, c, h * w
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)

        return k1 + k2


if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    cot = CoTAttention(in_planes=512, kernel_size=3)
    conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
    output = cot(input)
    print(output.shape)
    output = conv(input)
    print(output.shape)