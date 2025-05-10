"""
Copyright (c) 2025, Auorui.
All rights reserved.

AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE
    <https://arxiv.org/pdf/2010.11929.pdf>
Blog records: https://blog.csdn.net/m0_62919535/article/details/144936876
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from pyzjr.data.utils.tuplefun import to_2tuple
from pyzjr.nn.models.bricks.drop import DropPath

LayerNorm = partial(nn.LayerNorm, eps=1e-6)

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)

    def forward(self, x):
        x = self.proj(x) # 结果形状为 (batch_size, embed_dim, num_patches_H, num_patches_W)
        x = x.flatten(2) # 将输出展平成 (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2) # 转置为 (batch_size, num_patches, embed_dim)
        x = self.norm(x)
        return x

class MultiheadAttention(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads=8,
            qkv_bias=False,
            attention_dropout_ratio=0.,
            proj_drop=0.,
    ):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout_ratio)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[:3]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_linear(x)
        x = self.out_linear_drop(x)
        return x

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

class EncoderBlock(nn.Module):
    """Transformer encoder block.
    在 mlp block中, MLP 层的隐藏维度是输入的维度的4倍,
    详见 Table 1: Details of Vision Transformer model variants
    """
    mlp_ratio = 4
    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            drop_ratio=0.,
            attention_dropout_ratio=0.,
            drop_path_ratio=0.,
            norm_layer=LayerNorm,
            act_layer=nn.GELU
    ):
        super(EncoderBlock, self).__init__()
        self.num_heads = num_heads
        # Attention block
        self.norm1 = norm_layer(dim)
        self.attention = MultiheadAttention(dim, num_heads, qkv_bias=qkv_bias, attention_dropout_ratio=attention_dropout_ratio, proj_drop=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        # MLP block
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * self.mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, drop_ratio=drop_ratio, act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attention(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """堆叠 L 次 Transformer encoder block"""
    def __init__(
            self,
            num_layers,
            dim,
            num_heads,
            qkv_bias=False,
            drop_ratio=0.,
            attention_dropout_ratio=0.,
            drop_path_ratio=0.,
            norm_layer=LayerNorm,
            act_layer=nn.GELU
    ):
        super(TransformerEncoder, self).__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, num_layers)]  # stochastic depth decay rule
        self.layers = nn.ModuleList([
            EncoderBlock(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_ratio=drop_ratio,
                attention_dropout_ratio=attention_dropout_ratio,
                drop_path_ratio=dpr[_],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for _ in range(num_layers)
        ])
        self.norm = norm_layer(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=1000,
            hidden_dim=768,
            num_heads=12,
            num_layers=12,
            qkv_bias=True,
            drop_ratio=0.,
            attention_dropout_ratio=0.,
            drop_path_ratio=0.,
            norm_layer=LayerNorm,
            act_layer=nn.GELU
    ):
        super(VisionTransformer, self).__init__()
        assert img_size == 224, f"Image size must be 224, but got {img_size}"
        assert img_size % patch_size == 0, f"Image size {img_size} must be divisible by patch size {patch_size}"
        self.num_classes = num_classes
        self.num_tokens = 1
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels,
                                          embed_dim=hidden_dim, norm_layer=norm_layer)
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, hidden_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        self.blocks = TransformerEncoder(
            num_layers=num_layers,
            dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            drop_ratio=drop_ratio,
            attention_dropout_ratio=attention_dropout_ratio,
            drop_path_ratio=drop_path_ratio,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
        self.norm = norm_layer(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, x):
        x = self.patch_embed(x)  # [B, 196, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, 768]
        x = torch.cat((cls_token, x), dim=1)  # [B, 196+1, 768]
        x = self.pos_drop(x + self.pos_embed)  # [B, 197, 768]
        x = self.blocks(x)  # [B, 197, 768]
        x = x[:, 0]  # [B, 768]
        x = self.head(x)  # [B, num_classes]
        return x


def vit_b_16(num_classes=1000) -> VisionTransformer:
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        num_classes=num_classes,
        hidden_dim=768,
        num_heads=12,
        num_layers=12,
    )

def vit_b_32(num_classes=1000) -> VisionTransformer:
    return VisionTransformer(
        img_size=224,
        patch_size=32,
        num_classes=num_classes,
        hidden_dim=768,
        num_heads=12,
        num_layers=12,
    )

def vit_l_16(num_classes=1000) -> VisionTransformer:
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        num_classes=num_classes,
        hidden_dim=1024,
        num_heads=16,
        num_layers=24,
    )

def vit_l_32(num_classes=1000) -> VisionTransformer:
    return VisionTransformer(
        img_size=224,
        patch_size=32,
        num_classes=num_classes,
        hidden_dim=1024,
        num_heads=16,
        num_layers=24,
    )

def vit_h_14(num_classes=1000) -> VisionTransformer:
    return VisionTransformer(
        img_size=224,
        patch_size=14,
        num_classes=num_classes,
        hidden_dim=1280,
        num_heads=16,
        num_layers=32,
    )


if __name__=="__main__":
    import torchsummary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = vit_b_16(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    torchsummary.summary(net, input_size=(3, 224, 224))
    # vit_b_16 Total params: 85,651,204
    # vit_b_32 Total params: 87,420,676
    # vit_l_16 Total params: 303,105,028
    # vit_l_32 Total params: 305,464,324
    # vit_h_14 Total params: 630,442,244