"""
部分卷积 Partial convolution
Paper address: <https://arxiv.org/pdf/2303.03667.pdf>
Reference from: https://github.com/JierunChen/FasterNet/blob/master/models/fasternet.py
"""
import torch
import torch.nn as nn

class PartialConv(nn.Module):
    def __init__(self, dim, n_div, kernel_size=3, forward='split_cat'):
        """
        PartialConv 模块

        Args:
            dim (int): 输入张量的通道数。
            n_div (int): 分割通道数的分母，用于确定部分卷积的通道数。
            forward (str): 使用的前向传播方法，可选 'slicing' 或 'split_cat'。
        """
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    block = PartialConv(16, 4).to(device)
    input = torch.rand(1, 16, 64, 64).to(device)
    print(input.shape)
    output = block(input)
    print(output.shape)