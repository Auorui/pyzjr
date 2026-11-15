import torch.nn as nn

__all__ = ["xavier_init", "normal_init", "uniform_init", "trunc_normal_init"]


def xavier_init(module: nn.Module, gain: float = 1, bias: float = 0, distribution: str = 'normal') -> None:
    """Initialize the weights using Xavier (Glorot) initialization."""
    assert distribution in ['uniform', 'normal'], "distribution must be 'uniform' or 'normal'"
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module: nn.Module, mean: float = 0, std: float = 1, bias: float = 0) -> None:
    """Initialize the weights using a normal distribution."""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module: nn.Module, a: float = 0, b: float = 1, bias: float = 0) -> None:
    """Initialize the weights using a uniform distribution."""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def trunc_normal_init(module: nn.Module, mean: float = 0, std: float = 1, a: float = -2, b: float = 2, bias: float = 0) -> None:
    """Initialize the weights using a truncated normal distribution.
    Note: Requires PyTorch >= 1.7.0, which has built-in `nn.init.trunc_normal_`.
    """
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.trunc_normal_(module.weight, mean, std, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)