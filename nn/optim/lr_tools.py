"""
Small tool: used for learning rate range testing
"""
import torch.optim as optim
import torch.optim.lr_scheduler as torch_lr_scheduler
from pyzjr.nn.optim.adjust_lr import OnceCycleLR, PolyLR, CosineAnnealingLRWithDecay,\
    CosineAnnealingWarmRestartsWithDecay, WarmUpLR, WarmUpWithTrainloader

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_optimizer(model, optimizer_type='adamw', init_lr=0.001, momentum=None, weight_decay=None):
    """
    Returns a PyTorch optimizer with specified configuration and automatically handles common hyperparameters.

    Recommends using these official PyTorch optimizers for different scenarios:
    - Adam/AdamW: General purpose, transformer models, NLP tasks
    - SGD: Computer vision, GAN training, when using momentum
    - RMSprop: Recurrent networks, non-stationary objectives
    - Adagrad: Sparse data, recommendation systems
    - LBFGS: Small batch scenarios, physics-informed neural networks

    Args:
        model (nn.Module): Neural network model containing learnable parameters.
        optimizer_type (str): Type of optimizer. Supported values:
            'adam', 'sgd', 'adamw' (case-insensitive). Default: 'adamw'.
        init_lr (float): Initial learning rate. Default: 0.001.
        momentum (float, optional): Momentum value (used differently per optimizer):
            - For SGD: Traditional momentum (0.9 typical)
            - For Adam/AdamW: beta1 parameter (0.9 recommended)
            Defaults to None (uses optimizer-specific default).
        weight_decay (float, optional): L2 regularization strength.
            Defaults to None (uses optimizer-specific default: 1e-4 for Adam/SGD, 1e-2 for AdamW).

    Returns:
        Optimizer: Configured PyTorch optimizer instance.

    Raises:
        ValueError: If unsupported optimizer type is specified.

    Official PyTorch Optimizers Recommended:
        1. optim.RMSprop: For recurrent networks/non-stationary targets
        2. optim.Adadelta: For dynamic learning rate adaptation
        3. optim.NAdam: Adam variant with Nesterov momentum
        4. optim.RAdam: Rectified Adam with variance rectification
        5. optim.LBFGS: Memory-intensive but powerful for small batches
    """
    if optimizer_type in ['adam', 'Adam']:
        optimizer = optim.Adam(model.parameters(), lr=init_lr, betas=(momentum or 0.9, 0.999),
                               weight_decay=(weight_decay or 1e-4))

    elif optimizer_type in ['sgd', 'SGD']:
        optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=(momentum or 0.9), nesterov=True,
                              weight_decay=(weight_decay or 1e-4))

    elif optimizer_type in ['adamw', 'AdamW']:
        optimizer = optim.AdamW(model.parameters(), lr=init_lr, betas=(momentum or 0.9, 0.999),
                                weight_decay=weight_decay or 1e-2, eps=1e-8)

    elif optimizer_type in ['adagrad', 'Adagrad']:
        optimizer = optim.Adagrad(model.parameters(), lr=init_lr,
                                  weight_decay=weight_decay or 1e-6, lr_decay=1e-2)

    elif optimizer_type in ['rmsprop', 'RMSprop']:
        optimizer = optim.RMSprop(model.parameters(), lr=init_lr, momentum=momentum or 0.9,
                                  weight_decay=weight_decay or 1e-4, alpha=0.99)

    elif optimizer_type in ['adadelta', 'Adadelta']:
        optimizer = optim.Adadelta(model.parameters(), lr=init_lr,
                                   weight_decay=weight_decay or 1e-3, rho=0.9)

    elif optimizer_type in ['rprop', 'Rprop']:
        optimizer = optim.Rprop(model.parameters(), lr=init_lr,
                                etas=(0.5, 1.2), step_sizes=(1e-6, 50))
    else:
        raise ValueError("Unsupported optimizer type:: {}".format(optimizer_type))

    return optimizer

def get_lr_scheduler(optimizer, init_lr, scheduler_type='warmup', total_epochs=100,
                     milestones=None, trainloader=None, decay=True, gamma=1):
    """
    Factory function for creating learning rate schedulers with extended PyTorch support.

    Implements custom schedulers and integrates with official PyTorch schedulers.
    Recommended official schedulers included in this implementation:
    - StepLR/MultiStepLR: Basic step-wise decay
    - CosineAnnealingLR: Standard cosine decay
    - ReduceLROnPlateau: Metric-based dynamic adjustment
    - CyclicLR: Cyclical learning rates
    - ExponentialLR: Continuous exponential decay

    Args:
        optimizer (Optimizer): Wrapped optimizer
        init_lr (float): Initial learning rate
        scheduler_type (str): Type of scheduler from:
            ['step', 'multistep', '1cycle', 'poly', 'cos', 'cos_restart',
             'exponential', 'plateau', 'clr', 'warmup']
        total_epochs (int): Total training epochs for scheduling
        milestones (list): Epoch indices for MultiStepLR (auto-generated if None)
        trainloader (DataLoader): Required for warmup schedulers
        decay (bool): Whether to apply decay for cosine schedulers
        gamma (float): Base decay factor (interpretation varies per scheduler)

    Returns:
        _LRScheduler: Configured learning rate scheduler

    Official PyTorch Schedulers Recommended:
        1. CosineAnnealingWarmRestarts: Periodic restarts with cosine annealing
        2. OneCycleLR: Super-convergence schedule (better than custom 1cycle)
        3. LambdaLR: Fully customizable via lambda functions
        4. ChainedScheduler: Combine multiple schedulers sequentially
    """
    if milestones is None:
        # If milestones are not provided, distribute evenly
        milestones = [total_epochs // 5 * i for i in range(1, 5)]
    if scheduler_type in ['step']:
        scheduler = torch_lr_scheduler.StepLR(optimizer, step_size=milestones[0] * 2, gamma=gamma * 0.1)
    elif scheduler_type in ['multistep']:
        scheduler = torch_lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma * 0.1)
    elif scheduler_type in ['1cycle', 'oncecycle']:
        scheduler = OnceCycleLR(optimizer, total_epochs + 1, min_lr_factor=0.01, max_lr=3 * init_lr)
    elif scheduler_type in ['poly']:
        scheduler = PolyLR(optimizer, total_epochs, gamma=gamma * 0.9)
    elif scheduler_type in ['cos']:
        if decay:
            gamma = max(gamma - 0.01, 0.98)
        scheduler = CosineAnnealingLRWithDecay(optimizer, T_max=total_epochs, gamma=gamma,
                                               eta_min=init_lr * 1e-2)
        # if gamma == 1: scheduler = CosineAnnealingLR
    elif scheduler_type in ['cos_restart']:
        if decay:
            gamma = max(gamma - 0.02, 0.96)
        scheduler = CosineAnnealingWarmRestartsWithDecay(
            optimizer, T_0=total_epochs//2, gamma=gamma, eta_min=init_lr * 1e-2
        )
        # if gamma == 1 && decay=False: scheduler = CosineAnnealingWarmRestarts
    elif scheduler_type in ['exponential', 'exp']:
        scheduler = torch_lr_scheduler.ExponentialLR(optimizer, gamma=gamma - 0.05)
    elif scheduler_type in ['plateau']:
        scheduler = torch_lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
    elif scheduler_type in ['clr']:  # Cyclical Learning Rates
        scheduler = torch_lr_scheduler.CyclicLR(
            optimizer, base_lr=init_lr/10, max_lr=init_lr, step_size_up=2000, cycle_momentum=False
        )
    elif scheduler_type in ['warmup', 'warm']:
        if trainloader:
            scheduler = WarmUpWithTrainloader(optimizer, trainloader, total_epochs)
        else:
            scheduler = WarmUpLR(optimizer, int(total_epochs * 0.05))
    else:
        raise ValueError(f"Scheduler type '{scheduler_type}' is not supported.")

    return scheduler