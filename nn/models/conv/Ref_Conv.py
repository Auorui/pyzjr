"""
这种重参数化的重聚焦卷积（RefConv）方法是在已经有一个预训练的卷积神经网络（CNN）模型的基础上进行的。
具体来说，RefConv通过将可训练的重新聚焦变换应用于预训练模型的卷积核，以建立卷积核之间的连接，从而增强了模型的先验
Paper address: https://arxiv.org/pdf/2310.10563.pdf
The author provides the implementation of the pytorch algorithm at the end of the paper
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

class RefConv(nn.Module):
    """
    Implementation of RefConv.
        --in_channels: number of input channels in the basis kernel
        --out_channels: number of output channels in the basis kernel
        --kernel_size: size of the basis kernel
        --stride: stride of the original convolution
        --padding: padding added to all four sides of the basis kernel
        --groups: groups of the original convolution
        --map_k: size of the learnable kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding=None, groups=1, map_k=3):
        super(RefConv, self).__init__()
        assert map_k <= kernel_size
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
        G = in_channels * out_channels // (groups ** 2)
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2,
                                 groups=G, bias=False)

        self.bias = None
        self.stride = stride
        self.groups = groups
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

    def forward(self, inputs):
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding,
                        dilation=1, groups=self.groups, bias=self.bias)


if __name__ == '__main__':
    block1 = RefConv(64, 32, 3, 1)
    block2 = nn.Conv2d(64, 32, 3, 1, padding=1)
    input = torch.randn((3, 64, 64, 64))
    output = block1(input)
    print(output.shape)
    output = block2(input)
    print(output.shape)