import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Union

class ConvNormActivation(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple],
            stride: Union[int, tuple] = 1,
            padding: Union[int, tuple, str] = 0,
            dilation: Union[int, tuple] = 1,
            groups: int = 1,
            bias: bool = True,
            apply_act: bool = True,
            conv_layer: Callable[..., nn.Module] = nn.Conv2d,
            norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
            activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
            **kwargs
    ):
        super(ConvNormActivation, self).__init__()
        self.conv = conv_layer(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, **kwargs
        )
        layers = [self.conv]
        if self.norm is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None and apply_act:
            layers.append(activation_layer())
        self.block = nn.Sequential(*layers)

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        return self.block(x)

class AutopadConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=None, dilation=1,
                 groups=1, bias=True, padding_mode='reflect', **kwargs):
        k, p, d = kernel_size, padding, dilation
        if d > 1:
            k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
        if p is None:
            self.p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
        super(AutopadConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                            padding=self.p, dilation=dilation, groups=groups, bias=bias,
                                            padding_mode=padding_mode, **kwargs)

class DepthwiseConv2d(nn.Conv2d):
    # Depthwise Convolution
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, bias=True, **kwargs):
        super(DepthwiseConv2d, self).__init__(in_channels, in_channels, kernel_size,
                                       stride=stride, padding=padding, bias=bias, groups=in_channels, **kwargs)

class PointwiseConv2d(nn.Conv2d):
    # Pointwise Convolution
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super(PointwiseConv2d, self).__init__(in_channels, out_channels, 1, stride=1, padding=0, bias=bias, **kwargs)

class DepthwiseSeparableConv2d(nn.Module):
    # Depth-wise separable convolution
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True, **kwargs):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = DepthwiseConv2d(in_channels, kernel_size, stride, padding, bias, **kwargs)
        self.pointwise = PointwiseConv2d(in_channels, out_channels, bias, groups=groups)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class PartialConv2d(nn.Module):
    def __init__(self, dim, n_div, kernel_size=3, forward='split_cat'):
        """
        PartialConv
        code from: https://github.com/JierunChen/FasterNet/blob/master/models/fasternet.py
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


class LayerNorm(nn.Module):
    r""" LayerNorm implementation used in ConvNeXt
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", reshape_last_to_first=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        self.reshape_last_to_first = reshape_last_to_first

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x