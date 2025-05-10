"""
Copyright (c) 2025, Auorui.
All rights reserved.

Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/pdf/2103.14030>
use for reference: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer.py
                   https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/swin_transformer/model.py
"""
from functools import partial

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pyzjr.data.utils.tuplefun import to_2tuple
from pyzjr.nn.models.bricks.drop import DropPath
from pyzjr.nn.models.bricks.initer import trunc_normal_

LayerNorm = partial(nn.LayerNorm, eps=1e-6)

class PatchPartition(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, self.embed_dim,
                              kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        if H % self.patch_size[0] != 0:
            pad_h = self.patch_size[0] - H % self.patch_size[0]
            x = F.pad(x, (0, 0, 0, pad_h))
        if W % self.patch_size[1] != 0:
            pad_w = self.patch_size[1] - W % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, 0))
        x = self.proj(x)     # [B, embed_dim, H/patch_size, W/patch_size]
        Wh, Ww = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)    # [B, num_patches, embed_dim]
        # Linear Embedding
        x = self.norm(x)
        # x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x, Wh, Ww

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop_ratio=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_ratio)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports shifted and non-shifted windows.
    """
    def __init__(
            self,
            dim,
            window_size,
            num_heads,
            qkv_bias=True,
            proj_bias=True,
            attention_dropout_ratio=0.,
            proj_drop=0.,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        win_h, win_w = self.window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads)
        )   # [2*Wh-1 * 2*Ww-1, nHeads]   Offset Range: -Wh+1, Wh-1

        self.register_buffer("relative_position_index",
                             self.get_relative_position_index(win_h, win_w), persistent=False)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout_ratio)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def get_relative_position_index(self, win_h: int, win_w: int):
        # get pair-wise relative position index for each token inside the window
        coords = torch.stack(torch.meshgrid(torch.arange(win_h), torch.arange(win_w), indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
        relative_coords[:, :, 1] += win_w - 1
        relative_coords[:, :, 0] *= 2 * win_w - 1
        return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[:3]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block."""
    mlp_ratio = 4
    def __init__(
            self,
            dim,
            num_heads,
            window_size=7,
            shift_size=0,
            qkv_bias=True,
            proj_bias=True,
            attention_dropout_ratio=0.,
            proj_drop=0.,
            drop_path_ratio=0.,
            norm_layer=LayerNorm,
            act_layer=nn.GELU,
    ):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attention_dropout_ratio=attention_dropout_ratio,
            proj_drop=proj_drop,
        )

        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * self.mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_ratio=proj_bias)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage."""
    def __init__(self,
                 dim,
                 num_layers,
                 num_heads,
                 drop_path,
                 window_size=7,
                 qkv_bias=True,
                 proj_bias=True,
                 attention_dropout_ratio=0.,
                 proj_drop=0.,
                 norm_layer=LayerNorm,
                 act_layer=nn.GELU,
                 downsample=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.num_layers = num_layers

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attention_dropout_ratio=attention_dropout_ratio,
                proj_drop=proj_drop,
                drop_path_ratio=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                act_layer=act_layer)
            for i in range(num_layers)])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W




class SwinTransformer(nn.Module):
    """ Swin Transformer backbone."""
    def __init__(self,
                 patch_size=4,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=7,
                 qkv_bias=True,
                 proj_bias=True,
                 attention_dropout_ratio=0.,
                 proj_drop=0.,
                 drop_path_rate=0.2,
                 norm_layer=LayerNorm,
                 patch_norm=True,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # split image into non-overlapping patches
        self.patch_embed = PatchPartition(
            patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=proj_drop)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        layers = []
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                num_layers=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attention_dropout_ratio=attention_dropout_ratio,
                proj_drop=proj_drop,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
            )
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, L, C]
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # [B, L, C]
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

def swin_t(num_classes) -> SwinTransformer:
    model = SwinTransformer(in_channels=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes)
    return model

def swin_s(num_classes) -> SwinTransformer:
    model = SwinTransformer(in_channels=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 18, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes)
    return model


def swin_b(num_classes) -> SwinTransformer:
    model = SwinTransformer(in_channels=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes)
    return model

def swin_l(num_classes) -> SwinTransformer:
    model = SwinTransformer(in_channels=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=192,
                            depths=(2, 2, 18, 2),
                            num_heads=(6, 12, 24, 48),
                            num_classes=num_classes)
    return model

if __name__=="__main__":
    import pyzjr
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = swin_l(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    pyzjr.summary_1(net, input_size=(3, 224, 224))
    # swin_t Total params: 27,499,108
    # swin_s Total params: 48,792,676
    # swin_b Total params: 86,683,780
    # swin_l Total params: 194,906,308