"""
Copyright (c) 2023, Auorui.
All rights reserved.

reference <https://arxiv.org/pdf/2211.11943.pdf> (Conv2Former: A Simple Transformer-Style ConvNet for Visual Recognition)
Time:2023.12.31, Complete before the end of 2023.
"""
import torch
import torch.nn as nn

from pyzjr.Models.bricks import DropPath

__all__=["Conv2Former", "Conv2Former_n", "Conv2Former_t", "Conv2Former_s", "Conv2Former_b", "Conv2Former_l"]


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        return x


class ConvMod(nn.Module):
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



class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop_path=0.):
        super().__init__()

        self.attn = ConvMod(dim)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

class BaseLayer(nn.Module):
    def __init__(self, dim, depth, mlp_ratio=4., drop_path=None, downsample=True):
        super().__init__()
        self.dim = dim
        self.drop_path = drop_path

        self.blocks = nn.ModuleList([
            Block(dim=self.dim,mlp_ratio=mlp_ratio,drop_path=drop_path[i],)
            for i in range(depth)
        ])

        # patch merging layer
        if downsample:
            self.downsample = nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=dim),
                nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2,bias=False)
            )
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class Conv2Former(nn.Module):
    def __init__(self, num_classes=10, depths=(2,2,8,2), dim=(64,128,256,512), mlp_ratio=2.,drop_rate=0.,
                 drop_path_rate=0.15, **kwargs):
        super().__init__()

        norm_layer = nn.LayerNorm
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BaseLayer(dim[i_layer],
                              depth=depths[i_layer],
                              mlp_ratio=self.mlp_ratio,
                              drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                              downsample=(i_layer < self.num_layers - 1),
                              )
            self.layers.append(layer)
        self.fc1 = nn.Conv2d(3, dim[0], 1)
        self.norm = norm_layer(dim[-1], eps=1e-6,)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dim[-1], num_classes) \
            if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0.)
            nn.init.constant_(m.weight, 1.)

    def forward_features(self, x):
        x = self.fc1(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


#  reference <https://arxiv.org/pdf/2211.11943.pdf> Table 1
C = {'n': [64, 128, 256, 512],
     't': [72, 144, 288, 576],
     's': [72, 144, 288, 576],
     'b': [96, 192, 384, 768],
     'l': [128, 256, 512, 1024],
     }
L = {'n': [2, 2, 8, 2],
     't': [3, 3, 12, 3],
     's': [4, 4, 32, 4],
     'b': [4, 4, 34, 4],
     'l': [4, 4, 48, 4],
     }

def Conv2Former_n(num_classes, mlp_ratio=2, drop_path_rate=0.1):
    model = Conv2Former(num_classes=num_classes, depths=L["l"], dim=C["l"], mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate)
    return model

def Conv2Former_t(num_classes, mlp_ratio=2, drop_path_rate=0.1):
    model = Conv2Former(num_classes=num_classes, depths=L["t"], dim=C["t"], mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate)
    return model

def Conv2Former_s(num_classes, mlp_ratio=2, drop_path_rate=0.1):
    model = Conv2Former(num_classes=num_classes, depths=L["s"], dim=C["s"], mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate)
    return model

def Conv2Former_b(num_classes, mlp_ratio=2, drop_path_rate=0.1):
    model = Conv2Former(num_classes=num_classes, depths=L["b"], dim=C["b"], mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate)
    return model

def Conv2Former_l(num_classes, mlp_ratio=2, drop_path_rate=0.1):
    model = Conv2Former(num_classes=num_classes, depths=L["l"], dim=C["l"], mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate)
    return model

if __name__ == '__main__':
    model = Conv2Former(num_classes=10, depths=L["l"], dim=C["l"], mlp_ratio=2, drop_path_rate=0.1)

    input_tensor = torch.randn(1, 3, 224, 224)

    output = model(input_tensor)
    print("Output shape:", output.shape)




