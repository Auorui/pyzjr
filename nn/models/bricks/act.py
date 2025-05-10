"""
Copyright (c) 2024, Auorui.
All rights reserved.

这里仅仅提供了一个激活函数的常用类型选择, 更多的激活函数手写实现你可以从‘pyzjr.nn.torchutils.function’中了解
"""
import torch
import torch.nn as nn

class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)


class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(dim=1, **params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif name == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/clamp/relu/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)