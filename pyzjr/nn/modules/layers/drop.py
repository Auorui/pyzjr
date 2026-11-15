import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DropPath", "DropBlock"]


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    I follow the implementation:
        https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

    def drop_path(self, x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.
        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class DropBlock(nn.Module):
    """
    DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    Extremely fast and robust implementation adapted from `timm`.
    """
    def __init__(
            self,
            drop_prob: float = 0.1,
            block_size: int = 7,
            gamma_scale: float = 1.0,
            with_noise: bool = False,
            inplace: bool = False):
        super(DropBlock, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        return self.drop_block_fast(
            x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace)

    def drop_block_fast(
            self,
            x: torch.Tensor,
            drop_prob: float = 0.1,
            block_size: int = 7,
            gamma_scale: float = 1.0,
            with_noise: bool = False,
            inplace: bool = False,
    ):
        """
        DropBlock with an experimental gaussian noise option.
        Simplified from the original version without concern for valid block mask at edges.
        """
        B, C, H, W = x.shape
        total_size = W * H
        clipped_block_size = min(block_size, min(W, H))
        gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
                (W - block_size + 1) * (H - block_size + 1))

        block_mask = torch.empty_like(x).bernoulli_(gamma)
        block_mask = F.max_pool2d(
            block_mask.to(x.dtype), kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)

        if with_noise:
            normal_noise = torch.empty_like(x).normal_()
            if inplace:
                x.mul_(1. - block_mask).add_(normal_noise * block_mask)
            else:
                x = x * (1. - block_mask) + normal_noise * block_mask
        else:
            block_mask = 1 - block_mask
            normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-6)).to(dtype=x.dtype)
            if inplace:
                x.mul_(block_mask * normalize_scale)
            else:
                x = x * block_mask * normalize_scale
        return x