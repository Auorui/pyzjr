import torch
import random
import warnings
import numpy as np


def SeedEvery(seed=11, rank=0, use_deterministic_algorithms=None):
    """
    Set random seeds for Python, NumPy, PyTorch, and CUDA to ensure reproducibility.

    This function configures the environment so that every run produces the same results.
    It also handles CUDA-specific configurations (like cuDNN) and optionally enforces
    deterministic algorithms at the cost of performance.

    Args:
        seed (int, optional): The base random seed. Defaults to 11.
        rank (int, optional): The process rank for distributed training.
                              The actual seed used will be `seed + rank`. Defaults to 0.
        use_deterministic_algorithms (bool, optional): If True, forces PyTorch to use
                              deterministic algorithms. This may reduce performance.
                              Defaults to None.

    Returns:
        None
    """
    combined_seed = seed + rank
    random.seed(combined_seed)
    np.random.seed(combined_seed)
    torch.manual_seed(combined_seed)
    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        import torch.backends.cuda as cuda
        cuda.matmul.allow_tf32 = True
        cudnn.benchmark = True
        torch.cuda.manual_seed(combined_seed)
        torch.cuda.manual_seed_all(combined_seed)

    if torch.backends.flags_frozen():
        warnings.warn("PyTorch global flag support of backends is disabled, enable it to set global `cudnn` flags.")
        torch.backends.__allow_nonbracketed_mutation_flag = True
        
    if use_deterministic_algorithms:
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
        elif hasattr(torch, "set_deterministic"):
            torch.set_deterministic(True)
        else:
            warnings.warn("If use_deterministic-algorithms=True is set, pytorch version "
                          "greater than 1.8 may be required to use this feature properly.")







