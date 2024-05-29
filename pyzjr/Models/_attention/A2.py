"""
A2-Nets: Double Attention Networks
Original paper addresshttps: <https://arxiv.org/pdf/1810.11579.pdf>
"""
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class DoubleAttention(nn.Module):
    def __init__(self, in_channels, c_m, c_n, reconstruct=True):
        super().__init__()
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.c_m = c_m
        self.c_n = c_n
        self.convA = nn.Conv2d(in_channels, c_m, 1)
        self.convB = nn.Conv2d(in_channels, c_n, 1)
        self.convV = nn.Conv2d(in_channels, c_n, 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.in_channels
        A = self.convA(x)  # b, c_m, h, w
        B = self.convB(x)  # b, c_n, h, w
        V = self.convV(x)  # b, c_n, h, w
        tmp_a = A.view(b, self.c_m, -1)
        attention_maps = F.softmax(B.view(b, self.c_n, -1), dim=-1)
        attention_vectors = F.softmax(V.view(b, self.c_n, -1), dim=-1)
        # 第一步：特征提取
        global_descriptors = torch.bmm(tmp_a, attention_maps.permute(0, 2, 1))  # b.c_m, c_n
        # 第二步：特征分布
        tmp_z = global_descriptors.matmul(attention_vectors)  # b, c_m, h*w
        tmp_z = tmp_z.view(b, self.c_m, h, w)  # b,c_m,h,w
        if self.reconstruct:
            tmp_z = self.conv_reconstruct(tmp_z)

        return tmp_z


if __name__ == '__main__':
    block = DoubleAttention(64, 128, 128)
    input = torch.rand(1, 64, 64, 64)
    output = block(input)
    print(output.shape)