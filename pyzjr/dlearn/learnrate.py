# 学习率动态调整
import torch
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyzjr.core import _LRScheduler

def get_optimizer(model, optimizer_type='adam', init_lr=0.001, momentum=None, weight_decay=None):
    """
    根据指定的优化器类型返回相应的优化器对象，并根据批次大小调整初始学习率。

    :param model: 要优化的神经网络模型
    :param optimizer_type: 优化器类型，可以是 'adam' 或 'sgd'，默认为 'adam'
    :param init_lr: 初始学习率，默认为 0.001
    :param momentum: SGD优化器的动量参数，默认为 None
    :param weight_decay: 权重衰减（L2正则化）参数，默认为 None
    :return: 优化器对象
    """
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=init_lr, betas=(momentum or 0.9, 0.999),
                               weight_decay=(weight_decay or 1e-4))
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=(momentum or 0.9), nesterov=True,
                              weight_decay=(weight_decay or 1e-4))
    else:
        raise ValueError("Unsupported optimizer type:: {}".format(optimizer_type))

    return optimizer

class CustomScheduler:
    def __init__(self, optimizer, num_step, epochs, mode='cos', warmup=True, warmup_epochs=1, warmup_factor=1e-3, milestones=(30, 60, 90), gamma=0.1):
        """
        自定义学习率调度器，支持余弦退火（'cos'）和多步衰减（'step'）两种学习率调整模式。

        :param optimizer: 优化器对象，用于更新模型参数。
        :param num_step: 训练步数总数，通常等于 len(train_loader) * epochs。
        :param epochs: 总训练周期数。
        :param mode: 学习率调整模式，可以是 'cos'（余弦退火）或 'step'（多步衰减）。默认为 'cos'。
        :param warmup: 是否使用学习率预热。默认为 True。
        :param warmup_epochs: 学习率预热的周期数。仅在 warmup 为 True 时有效。默认为 1。
        :param warmup_factor: 学习率预热的初始倍率因子。仅在 warmup 为 True 时有效。默认为 1e-3。
        :param milestones: 多步衰减模式下的学习率降低的里程碑 epoch 列表。仅在 mode 为 'step' 时有效。默认为 (30, 60, 90)。
        :param gamma: 多步衰减模式下的学习率降低倍率因子。仅在 mode 为 'step' 时有效。默认为 0.1。
        """
        self.optimizer = optimizer
        self.num_step = num_step
        self.epochs = epochs
        self.use_multi_step = False
        if warmup is False:
            warmup_epochs = 0
        if mode == "cos":
            self.warmup = warmup
            self.warmup_epochs = warmup_epochs
            self.warmup_factor = warmup_factor
        elif mode == "step":
            self.use_multi_step = True
            self.milestones = milestones
            self.gamma = gamma
        self.scheduler = self.get_scheduler()

    def get_scheduler(self):
        assert self.num_step > 0 and self.epochs > 0
        if self.use_multi_step:
            return optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones, gamma=self.gamma)
        else:
            def f(x):
                """
                根据step数返回一个学习率倍率因子，
                """
                if self.warmup is True and x <= (self.warmup_epochs * self.num_step):
                    alpha = float(x) / (self.warmup_epochs * self.num_step)

                    return self.warmup_factor * (1 - alpha) + alpha
                else:
                    # warmup后lr倍率因子从1 -> 0
                    # 参考deeplab_v2: Learning rate policy
                    return (1 - (x - self.warmup_epochs * self.num_step) / ((self.epochs - self.warmup_epochs) * self.num_step)) ** 0.9

            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=f)

    def step(self):
        self.scheduler.step()

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class FindLR(_LRScheduler):
    """
    exponentially increasing learning rate

    Args:
        optimizer: optimzier(e.g. SGD)
        num_iter: totoal_iters
        max_lr: maximum  learning rate
    """
    def __init__(self, optimizer, max_lr=10, num_iter=100, last_epoch=-1):

        self.total_iters = num_iter
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (self.max_lr / base_lr) ** (self.last_epoch / (self.total_iters + 1e-32)) for base_lr in self.base_lrs]


class WarmUpLR(_LRScheduler):
    """
    warmup_training learning rate scheduler
    Args:
        optimizer: optimzier
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class lr_finder():
    def __init__(self,net, dataloader, loss_function,optimizer_type="sgd",num_iter=100,batch_size=4):
        self.net = net
        self.dataloader = dataloader
        self.loss_function = loss_function
        self.optimizer_type = optimizer_type
        self.num_iter = num_iter
        self.batch_size = batch_size

    def update(self, init_lr=1e-7, max_lr=10):
        n = 0
        learning_rate = []
        losses = []
        optimizer = get_optimizer(self.net, self.optimizer_type, init_lr)
        lr_scheduler = FindLR(optimizer, max_lr=max_lr, num_iter=self.num_iter)
        epoches = int(self.num_iter / len(self.dataloader)) + 1

        for epoch in range(epoches):
            self.net.train()
            for batch_index, (images, labels) in enumerate(self.dataloader):
                if n > self.num_iter:
                    break
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()
                predicts = self.net(images)
                loss = self.loss_function(predicts, labels)
                if torch.isnan(loss).any():
                    n += 1e8
                    break
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                print('Iterations: {iter_num} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.8f}'.format(
                    loss.item(),
                    optimizer.param_groups[0]['lr'],
                    iter_num=n,
                    trained_samples=batch_index * self.batch_size + len(images),
                    total_samples=len(self.dataloader),
                ))

                learning_rate.append(optimizer.param_groups[0]['lr'])
                losses.append(loss.item())
                n += 1

        self.learning_rate = learning_rate[10:-5]
        self.losses = losses[10:-5]

    def plotshow(self, show=True):
        import matplotlib
        matplotlib.use("TkAgg")
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.learning_rate, self.losses)
        ax.set_xlabel('learning rate')
        ax.set_ylabel('losses')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        if show:
            plt.show()

    def save(self, path='result.jpg'):
        self.plotshow(show=False)
        plt.savefig(path)
