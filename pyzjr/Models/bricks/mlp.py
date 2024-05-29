"""
MLP module w/ dropout and configurable activation layer
"""
import torch
import torch.nn as nn
from pyzjr.core import to_2tuple
from functools import partial

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GluMlp(nn.Module):
    """
    MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Sigmoid, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)
        self.init_weights()

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x, gates = x.chunk(2, dim=-1)
        x = x * self.act(gates)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GatedMlp(nn.Module):
    """MLP as used in gMLP"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 gate_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2  # FIXME base reduction on gate property?
        else:
            self.gate = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.gate(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMlp(nn.Module):
    """
    MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, norm_layer=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class ConvMod(nn.Module):
    """come from pyzjr.backbone.conv2former"""
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)
        return x


class GlobalResponseNormMlp(nn.Module):
    """ MLP w/ Global Response Norm (see grn.py), nn.Linear or 1x1 Conv2d
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.grn = GlobalResponseNorm(hidden_features, channels_last=not use_conv)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.grn(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GlobalResponseNorm(nn.Module):
    """ Global Response Normalization layer
    """
    def __init__(self, dim, eps=1e-6, channels_last=True):
        super().__init__()
        self.eps = eps
        if channels_last:
            self.spatial_dim = (1, 2)
            self.channel_dim = -1
            self.wb_shape = (1, 1, 1, -1)
        else:
            self.spatial_dim = (2, 3)
            self.channel_dim = 1
            self.wb_shape = (1, -1, 1, 1)

        self.weight = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)