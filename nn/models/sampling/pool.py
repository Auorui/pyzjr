"""
Copyright (c) 2024, Auorui.
All rights reserved.

The Torch implementation of average pooling and maximum pooling has been compared with the official Torch implementation
Time: 2024-01-22  17:28
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyzjr.data.utils.tuplefun import to_2tuple, to_4tuple

class MaxPool2d(nn.Module):
    """
    定义MaxPool2d类，结合了普通池化和矢量化池化的功能
    池化层计算公式:
        output_size = [(input_size−kernel_size) // stride + 1]
    """
    def __init__(self, kernel_size, stride, is_vectoring=True):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.is_vectoring = is_vectoring

    def max_pool2d_traditional(self, input_tensor):
        batch_size, channels, height, width = input_tensor.size()
        output_height = (height - self.kernel_size) // self.stride + 1
        output_width = (width - self.kernel_size) // self.stride + 1
        output_tensor = torch.zeros(batch_size, channels, output_height, output_width)

        for i in range(output_height):
            for j in range(output_width):
                window = input_tensor[:, :,
                         i * self.stride: i * self.stride + self.kernel_size, j * self.stride: j * self.stride + self.kernel_size]
                output_tensor[:, :, i, j] = torch.max(window.reshape(batch_size, channels, -1), dim=2)[0]
        return output_tensor

    def max_pool2d_vectorized(self, input_tensor):
        unfolded = input_tensor.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        max_values, _ = unfolded.max(dim=-1)
        max_values, _ = max_values.max(dim=-1)
        return max_values

    def forward(self, input_tensor):
        if self.is_vectoring:
            return self.max_pool2d_vectorized(input_tensor)
        else:
            return self.max_pool2d_traditional(input_tensor)


class AvgPool2d(nn.Module):
    """
    定义AvgPool2d类，结合了普通池化和矢量化池化的功能
    池化层计算公式:
        output_size = [(input_size−kernel_size) // stride + 1]
    """
    def __init__(self, kernel_size, stride, is_vectoring=True):
        super(AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.is_vectoring = is_vectoring

    def avg_pool2d_traditional(self, input_tensor):
        batch_size, channels, height, width = input_tensor.size()
        output_height = (height - self.kernel_size) // self.stride + 1
        output_width = (width - self.kernel_size) // self.stride + 1
        output_tensor = torch.zeros(batch_size, channels, output_height, output_width)

        for i in range(output_height):
            for j in range(output_width):
                window = input_tensor[:, :,
                         i * self.stride: i * self.stride + self.kernel_size, j * self.stride: j * self.stride + self.kernel_size]
                output_tensor[:, :, i, j] = torch.mean(window.reshape(batch_size, channels, -1), dim=2)
        return output_tensor

    def avg_pool2d_vectorized(self, input_tensor):
        unfolded = input_tensor.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        avg_values = unfolded.mean(dim=-1)
        avg_values = avg_values.mean(dim=-1)
        return avg_values

    def forward(self, input_tensor):
        if self.is_vectoring:
            return self.avg_pool2d_vectorized(input_tensor)
        else:
            return self.avg_pool2d_traditional(input_tensor)

class StridedPool2d(nn.Module):
    """
    实现一个跨步卷积层Strided Convolution, 本质上可以实现类似于池化操作的效果。
    通过步幅stride大于 1 的卷积操作来实现空间分辨率的降低。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, is_relu=True):
        super(StridedPool2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.is_relu = is_relu

    def forward(self, x):
        x = self.conv(x)
        if self.is_relu:
            x = self.relu(x)
        return x

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    Hacked together by / Copyright 2020 Ross Wightman
    Currently, it is rarely used in CNN, and the code is taken
    from the project pytorch-image-models

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.padding = to_4tuple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

if __name__=="__main__":
    # MaxPool2d与AvgPool2d手写测试实验
    # input_data = torch.rand((1, 3, 3, 3))
    # input_data = torch.Tensor([[[[0.3939, 0.8964, 0.3681],
    #                            [0.5134, 0.3780, 0.0047],
    #                            [0.0681, 0.0989, 0.5962]],
    #                           [[0.7954, 0.4811, 0.3329],
    #                            [0.8804, 0.3986, 0.3561],
    #                            [0.2797, 0.3672, 0.6508]],
    #                           [[0.6309, 0.1340, 0.0564],
    #                            [0.3101, 0.9927, 0.5554],
    #                            [0.0947, 0.2305, 0.8299]]]])
    #
    # print(input_data.shape)
    # is_vectoring = False
    # kernel_size = 3
    # stride = 2
    # MaxPool2d1 = nn.MaxPool2d(kernel_size, stride)
    # output_data_with_torch_max = MaxPool2d1(input_data)
    # AvgPool2d1 = nn.AvgPool2d(kernel_size, stride)
    # output_data_with_torch_avg = AvgPool2d1(input_data)
    # AvgPool2d2 = AvgPool2d(kernel_size, stride, is_vectoring)
    # output_data_with_torch_Avg = AvgPool2d2(input_data)
    # MaxPool2d2 = MaxPool2d(kernel_size, stride, is_vectoring)
    # output_data_with_torch_Max = MaxPool2d2(input_data)
    # # output_data_with_max = max_pool2d(input_data, kernel_size, stride)
    # # output_data_with_avg = avg_pool2d(input_data, kernel_size, stride)
    #
    # print("\ntorch.nn pooling Output:")
    # print(output_data_with_torch_max,"\n",output_data_with_torch_max.size())
    # print(output_data_with_torch_avg,"\n",output_data_with_torch_avg.size())
    # print("\npooling Output:")
    # print(output_data_with_torch_Max,"\n",output_data_with_torch_Max.size())
    # print(output_data_with_torch_Avg,"\n",output_data_with_torch_Avg.size())
    # # 直接使用bool方法判断会因为浮点数的原因出现偏差
    # print(torch.allclose(output_data_with_torch_max,output_data_with_torch_Max))
    # print(torch.allclose(output_data_with_torch_avg,output_data_with_torch_Avg))
    # tensor([[[[0.8964]],       # output_data_with_max
    #          [[0.8804]],
    #          [[0.9927]]]])
    # tensor([[[[0.3686]],       # output_data_with_avg
    #           [[0.5047]],
    #           [[0.4261]]]])
    #
    input_data = torch.rand((1, 3, 64, 64))
    strided_conv = StridedPool2d(3, 64)
    output_data = strided_conv(input_data)
    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)