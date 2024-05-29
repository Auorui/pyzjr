import torch
from collections import defaultdict, deque
import time
import datetime
from pyzjr.dlearn.tools import LoadingBar

__all__ = ["AverageMeter", "SmoothedValue", "MetricLogger", "ConfusionMatrix"]

class AverageMeter(object):
    """A simple class that maintains the running average of a quantity

    Example:
    ```
        loss_avg = AverageMeter()
        loss_avg.update(2)
        loss_avg.update(4)
        loss_avg() = 3
    ```
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t", load_bar = False):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'

        load_bar = LoadingBar(20)

        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'Process: {Process}',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'Process: {Process}',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    progress = (i + 1) / len(iterable)
                    bar_string = load_bar(progress)
                    print("\r", log_msg.format(
                        i,
                        len(iterable),
                        Process=bar_string,
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB), end=' ', flush=True)
                else:
                    progress = (i + 1) / len(iterable)
                    bar_string = load_bar(progress)
                    print("\r", log_msg.format(
                        i, bar_string,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)), end=' ', flush=True)
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("\n", f'{header} Total time: {total_time_str}')

class ConfusionMatrix(object):
    """For details:https://blog.csdn.net/m0_62919535/article/details/132893016"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, true, pred):
        t, p = true.flatten(), pred.flatten()
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=t.device)
        with torch.no_grad():
            k = (t >= 0) & (t < n)
            inds = n * t[k].to(torch.int64) + p[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    @property
    def ravel(self):
        """
        计算混淆矩阵的TN, FP, FN, TP
        支持二分类和多分类
        """
        h = self.mat.float()
        n = self.num_classes
        if n == 2:
            TP, FN, FP, TN = h.flatten()
            return TP, FN, FP, TN
        if n > 2:
            TP = h.diag()
            FN = h.sum(dim=1) - TP
            FP = h.sum(dim=0) - TP
            TN = torch.sum(h) - (torch.sum(h, dim=0) + torch.sum(h, dim=1) - TP)

            return TP, FN, FP, TN

    def compute(self):
        """
        主要在eval的时候使用,你可以调用ravel获得TN, FP, FN, TP, 进行其他指标的计算
        计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        计算每个类别的准确率
        计算每个类别预测与真实目标的iou,IoU = TP / (TP + FP + FN)
        """
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
            acc_global.item() * 100,
            ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100)
