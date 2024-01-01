import torch
import torch.nn as nn
import torch.nn.functional as F

class CBLR(nn.Sequential):
    """
    CBLR -> Conv+BN+LeakyReLU , Originating from darknet53
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, negative_slope=0.1):
        super(CBLR, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope)
        )

class ConvBnAct(nn.Module):
    """
    ConvBnAct -> Conv+Bn+Act(可选)  默认是ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, act_layer=nn.ReLU, inplace=True):
        super(ConvBnAct, self).__init__(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        act_layer(inplace=inplace)
        )


