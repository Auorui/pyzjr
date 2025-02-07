"""
Copyright (c) 2024, Auorui.
All rights reserved.

Supports one hot encoding for np and torch.
https://blog.csdn.net/m0_62919535/article/details/135516163?spm=1001.2014.3001.5501

time 2024-01-11
"""
import torch
import numpy as np
from pyzjr.utils.check import is_tensor, is_numpy

__all__ = [
    "one_hot",
    "get_one_hot",
    "get_one_hot_with_torch",
    "get_one_hot_with_np",
]

def one_hot(labels, num_classes=-1):
    """
    将标签转为独热编码, 经过测试与torch.nn.functional里面的函数测试相同
    :param labels: 标签
    :param num_classes: 默认为-1, 表示进行自动计算类别最大的那个
    Examples:
        >>> label_1 = torch.arange(0, 5) % 3
        tensor([0, 1, 2, 0, 1])
        >>> label_2 = torch.arange(0, 6).view(3, 2) % 3
        tensor([[0, 1], [2, 0], [1, 2]])
        >>> print(one_hot(label_1))
        tensor([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]])
        >>> print(one_hot(label_1, 5))
        tensor([[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0]])
        >>> print(one_hot(label_2))
        tensor([[[1, 0, 0],
                 [0, 1, 0]],
                [[0, 0, 1],
                 [1, 0, 0]],
                [[0, 1, 0],
                 [0, 0, 1]]])
    """
    if num_classes == -1:
        num_classes = int(labels.max()) + 1
    one_hot_tensor = torch.zeros(labels.size() + (num_classes,), dtype=torch.int64)
    one_hot_tensor.scatter_(-1, labels.unsqueeze(-1).to(torch.int64), 1)
    return one_hot_tensor

def get_one_hot_with_torch(labels, num_classes=-1):
    """用于分割网络的one hot"""
    labels = torch.as_tensor(labels)
    ones = one_hot(labels, num_classes)
    return ones.view(*labels.size(), num_classes)

def get_one_hot_with_np(labels, num_classes=-1):
    """用于处理标签还没有转为tensor的情况"""
    if num_classes == -1:
        num_classes = int(np.max(labels)) + 1
    input_shape = labels.shape
    seg_labels = np.eye(num_classes)[labels.reshape([-1])]
    seg_labels = seg_labels.reshape((int(input_shape[0]), int(input_shape[1]), num_classes))
    return seg_labels

def get_one_hot(labels, num_classes=-1):
    if is_numpy(labels):
        return get_one_hot_with_np(labels=labels, num_classes=num_classes)
    if is_tensor(labels):
        return get_one_hot_with_torch(labels=labels, num_classes=num_classes)

if __name__=="__main__":
    seg_labels = np.random.randint(0, 3, size=[512, 512])
    seg_labels2 = torch.randint(0, 3, size=[512, 512])
    print(get_one_hot(seg_labels))
    print(get_one_hot(seg_labels).shape)
    print(get_one_hot(seg_labels2))   # torch.Size([512, 512, 3])
    print(get_one_hot(seg_labels2).shape)