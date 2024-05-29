"""
Copyright (c) 2024, Auorui.
All rights reserved.

The Torch implementation of average pooling and maximum pooling has been compared with the official Torch implementation
Time: 2024-01-22  17:28
"""
import torch
import torch.nn as nn

class MaxPool2d(nn.Module):
    """
    池化层计算公式:
        output_size = [(input_size−kernel_size) // stride + 1]
    """
    def __init__(self, kernel_size, stride):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def max_pool2d(self, input_tensor, kernel_size, stride):
        batch_size, channels, height, width = input_tensor.size()
        output_height = (height - kernel_size) // stride + 1
        output_width = (width - kernel_size) // stride + 1
        output_tensor = torch.zeros(batch_size, channels, output_height, output_width)

        for i in range(output_height):
            for j in range(output_width):
                # 获取输入张量中与池化窗口对应的部分
                window = input_tensor[:, :,
                         i * stride: i * stride + kernel_size, j * stride: j * stride + kernel_size]
                output_tensor[:, :, i, j] = torch.max(window.reshape(batch_size, channels, -1), dim=2)[0]
        return output_tensor

    def forward(self, input_tensor):
        return self.max_pool2d(input_tensor, kernel_size=self.kernel_size, stride=self.stride)


class AvgPool2d(nn.Module):
    """
    池化层计算公式:
        output_size = [(input_size−kernel_size) // stride + 1]
    """
    def __init__(self, kernel_size, stride):
        super(AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def avg_pool2d(self, input_tensor, kernel_size, stride):
        batch_size, channels, height, width = input_tensor.size()
        output_height = (height - kernel_size) // stride + 1
        output_width = (width - kernel_size) // stride + 1
        output_tensor = torch.zeros(batch_size, channels, output_height, output_width)

        for i in range(output_height):
            for j in range(output_width):
                # 获取输入张量中与池化窗口对应的部分
                window = input_tensor[:, :,
                         i * stride: i * stride + kernel_size, j * stride:j * stride + kernel_size]
                output_tensor[:, :, i, j] = torch.mean(window.reshape(batch_size, channels, -1), dim=2)
        return output_tensor

    def forward(self, input_tensor):
        return self.avg_pool2d(input_tensor, kernel_size=self.kernel_size, stride=self.stride)


if __name__=="__main__":
    # input_data = torch.rand((1, 3, 3, 3))
    input_data = torch.Tensor([[[[0.3939, 0.8964, 0.3681],
                               [0.5134, 0.3780, 0.0047],
                               [0.0681, 0.0989, 0.5962]],
                              [[0.7954, 0.4811, 0.3329],
                               [0.8804, 0.3986, 0.3561],
                               [0.2797, 0.3672, 0.6508]],
                              [[0.6309, 0.1340, 0.0564],
                               [0.3101, 0.9927, 0.5554],
                               [0.0947, 0.2305, 0.8299]]]])

    print(input_data.shape)

    kernel_size = 3
    stride = 1
    MaxPool2d1 = nn.MaxPool2d(kernel_size, stride)
    output_data_with_torch_max = MaxPool2d1(input_data)
    AvgPool2d1 = nn.AvgPool2d(kernel_size, stride)
    output_data_with_torch_avg = AvgPool2d1(input_data)
    AvgPool2d2 = AvgPool2d(kernel_size, stride)
    output_data_with_torch_Avg = AvgPool2d2(input_data)
    MaxPool2d2 = MaxPool2d(kernel_size, stride)
    output_data_with_torch_Max = MaxPool2d2(input_data)
    # output_data_with_max = max_pool2d(input_data, kernel_size, stride)
    # output_data_with_avg = avg_pool2d(input_data, kernel_size, stride)

    print("\ntorch.nn pooling Output:")
    print(output_data_with_torch_max,"\n",output_data_with_torch_max.size())
    print(output_data_with_torch_avg,"\n",output_data_with_torch_avg.size())
    print("\npooling Output:")
    print(output_data_with_torch_Max,"\n",output_data_with_torch_Max.size())
    print(output_data_with_torch_Avg,"\n",output_data_with_torch_Avg.size())
    # 直接使用bool方法判断会因为浮点数的原因出现偏差
    print(torch.allclose(output_data_with_torch_max,output_data_with_torch_Max))
    print(torch.allclose(output_data_with_torch_avg,output_data_with_torch_Avg))
    # tensor([[[[0.8964]],       # output_data_with_max
    #          [[0.8804]],
    #          [[0.9927]]]])
    # tensor([[[[0.3686]],       # output_data_with_avg
    #           [[0.5047]],
    #           [[0.4261]]]])