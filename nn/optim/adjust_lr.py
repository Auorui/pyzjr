"""
Copyright (c) 2024, Auorui.
All rights reserved.
time 2024-05-25
"""
# https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
import math
import torch
import numpy as np
import torch.optim as optim
from pyzjr.nn.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts

class OnceCycleLR(_LRScheduler):
    def __init__(self, optimizer, epochs, min_lr_factor=0.05, max_lr=1.0):
        """
        Implements a one-time cycle learning rate schedule with three phases:
                          growth, decline, and decay.

        The learning rate first linearly increases from `min_lr_factor`*base_lr to `max_lr`,
        then linearly decreases to `min_lr_factor`*base_lr, followed by a final decay phase.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            epochs (int): Total number of training epochs.
            min_lr_factor (float, optional): Minimum learning rate factor relative to base learning rate. Defaults to 0.05.
            max_lr (float, optional): Absolute maximum learning rate (potentially overrides base_lr). Defaults to 1.0.

        The schedule divides training into three phases:
            1. Growth phase: First half of total epochs, linear growth from min_lr to max_lr
            2. Decline phase: (epochs - half_epochs - decay_epochs) epochs, linear decline back to min_lr
            3. Decay phase: Final 5% of epochs, linear decay to 1% of min_lr
        """
        half_epochs = epochs // 2
        decay_epochs = int(epochs * 0.05)

        lr_grow = np.linspace(min_lr_factor, max_lr, num=half_epochs)
        lr_down = np.linspace(max_lr, min_lr_factor, num=int(epochs - half_epochs - decay_epochs))
        lr_decay = np.linspace(min_lr_factor, min_lr_factor * 0.01, int(decay_epochs))
        self.learning_rates = np.concatenate((lr_grow, lr_down, lr_decay)) / max_lr
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * self.learning_rates[self.last_epoch] for base_lr in self.base_lrs]


class PolyLR(LambdaLR):
    """
    Implements polynomial learning rate decay based on epoch count.
    The learning rate is computed as: base_lr * (1 - epoch/max_epoch)^gamma

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_epoch (int): Maximum epoch count used for decay calculation.
        gamma (float, optional): Exponent factor for polynomial decay. Defaults to 0.9.
    """
    def __init__(self, optimizer, max_epoch, gamma=0.9):
        def poly_lr(epoch):
            return (1.0 - float(epoch) / max_epoch) ** gamma

        super().__init__(optimizer, poly_lr)

class CosineAnnealingLRWithDecay(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """
    def __init__(self, optimizer, T_max, gamma, eta_min=0, last_epoch=-1):
        self.gamma = gamma
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLRWithDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        def compute_lr(base_lr):
            return (
                    self.eta_min
                    + (base_lr * self.gamma**self.last_epoch - self.eta_min)
                    * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
                    / 2
            )

        return [compute_lr(base_lr) for base_lr in self.base_lrs]

class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose="deprecated"):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


class CosineAnnealingWarmRestartsWithDecay(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, gamma=0.9):
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)
        self.gamma = gamma

    def get_lr(self):
        return [
            self.eta_min
            + (base_lr * self.gamma**self.last_epoch - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            for base_lr in self.base_lrs
        ]

class WarmUpLR(_LRScheduler):
    """Implements linear learning rate warmup strategy during initial training phase.

    Gradually increases learning rate from 0 to base learning rate over specified iterations.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_iters (int): Total iterations for warmup phase (typically 1%-10% of total training steps).
        last_epoch (int, optional): The index of last epoch. Defaults to -1.

    Attributes:
        total_iters (int): Total warmup iterations stored during initialization.

    The learning rate grows linearly:
    `lr = base_lr * current_iteration / total_iters` during warmup phase.
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class WarmUpWithTrainloader(_LRScheduler):
    def __init__(self, optimizer, train_loader, epochs, warmup_epochs=1, warmup_factor=1e-3):
        """Implements hybrid warmup + polynomial decay learning rate schedule based on training steps.

        Combines two phases:
        1. Warmup: Linearly increases LR from `base_lr * warmup_factor` to `base_lr`
        2. Polynomial decay: Gradually decreases LR after warmup phase

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            train_loader (DataLoader): Training data loader for step calculation.
            epochs (int): Total number of training epochs.
            warmup_epochs (int, optional): Number of epochs for warmup phase. Defaults to 1.
            warmup_factor (float, optional): Initial LR multiplier during warmup. Defaults to 1e-3.

        Attributes:
            num_step (int): Total training steps (len(train_loader) * epochs).
            scheduler (LambdaLR): Internal lambda function-based scheduler.
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.num_step = len(train_loader) * self.epochs   # num_step训练步数总数，通常等于 len(train_loader) * epochs。
        self.warmup_factor = warmup_factor
        self.scheduler = self.get_scheduler()
        super().__init__(optimizer, last_epoch=-1)

    def get_scheduler(self):
        def f(x):
            if self.warmup_epochs > 0 and x <= (self.warmup_epochs * self.num_step):
                alpha = float(x) / (self.warmup_epochs * self.num_step)
                return self.warmup_factor * (1 - alpha) + alpha
            else:
                # warmup后lr倍率因子从1 -> 0
                # 参考deeplab_v2: Learning rate policy
                return (1 - (x - self.warmup_epochs * self.num_step) / ((self.epochs - self.warmup_epochs) * self.num_step)) ** 0.9
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=f)

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

def plot_lr_scheduler(model, criterion, optimizer, scheduler, epochs=100, label="1cycle"):
    from pyzjr.visualize import matplotlib_patch
    matplotlib_patch()
    import matplotlib.pyplot as plt
    x_data = torch.tensor([[1.0], [2.0], [3.0]])
    y_data = torch.tensor([[2.0], [4.0], [6.0]])
    plt.figure()
    lrs = []
    for epoch in range(epochs):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

    with torch.no_grad():
        test_data = torch.tensor([[4.0]])
        predicted_y = model(test_data)
        print(f'After training, input 4.0, predicted output: {predicted_y.item()}')
    plt.plot(range(epochs), lrs, label=label)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    from torch.optim import SGD
    import torch.optim.lr_scheduler as torch_lr_scheduler

    lr = 1e-3
    net = torch.nn.Linear(1, 1)
    opt = SGD(net.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    epochs = 100

    # label = "1cycle"
    # scheduler = OnceCycleLR(opt, epochs + 1, min_lr_factor=0.01, max_lr=3 * lr)

    # label = "poly"
    # scheduler = PolyLR(opt, epochs, gamma=0.9)

    # label = "cos decay"
    # scheduler = torch_lr_scheduler.CosineAnnealingLR(opt, epochs // 5, eta_min=lr * 1e-2)
    # scheduler = CosineAnnealingLRWithDecay(opt, epochs // 5, gamma=0.99, eta_min=lr * 1e-2)

    # label = "warm up"
    # scheduler = WarmUpLR(opt, epochs * 0.05)

    # label = "warm up with tr loader"
    # scheduler = WarmUpWithTrainloader(opt, [i for i in range(200)], epochs)

    # label = "cos warm restarts"
    # scheduler = torch_lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=epochs//2, T_mult=1, eta_min=lr * 1e-2)
    # scheduler = CosineAnnealingWarmRestartsWithDecay(opt, epochs//2, eta_min=lr * 1e-2, gamma=0.99)

    label = "step lr"
    # scheduler = torch_lr_scheduler.StepLR(opt, epochs//3)
    scheduler =torch_lr_scheduler.MultiStepLR(opt, milestones=[30, 80], gamma=0.1)
    plot_lr_scheduler(net, criterion, opt, scheduler, epochs, label=label)