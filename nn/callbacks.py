import os
import torch
from collections import deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal as signal
from torch.utils.tensorboard import SummaryWriter

__all__ = ["AverageMeter", "SmoothedValue", "EvalMetrics",
           "LossHistory", "ErrorRateMonitor", "ProcessMonitor"]

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


class EvalMetrics(object):
    """
    Segmentation evaluation metrics (IoU, Accuracy, etc.)
    For details:https://blog.csdn.net/m0_62919535/article/details/132893016
    """
    def __init__(self, num_classes, eps=1e-5):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.eps = eps

    def _fast_hist(self, label_pred, label_true, num_classes):
        mask = (label_true >= 0) & (label_true < num_classes)
        hist = np.bincount(
            num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
        return hist

    def update(self, pred, true):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(true, torch.Tensor):
            true = true.cpu().numpy()
        for lp, lt in zip(pred, true):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten(), self.num_classes)

    def compute(self, hist):
        acc = np.diag(hist).sum() / (hist.sum() + self.eps)
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + self.eps)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + self.eps)
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, mean_iu, fwavacc

    def reset(self):
        """Reset all accumulated state"""
        self.hist = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def eval(self):
        """Print formatted evaluation metrics"""
        acc, acc_cls, mean_iu, fwavacc = self.compute(self.hist)
        print(
            f'Global Accuracy: {acc:.2%}\n'
            f'Class Accuracy: {acc_cls:.2%}\n'
            f'Mean IoU: {mean_iu:.2%}\n'
            f'Weighted IoU: {fwavacc:.2%}'
        )

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []

        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        try:
            device = 'cpu'
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1]).to(device)
            self.writer.add_graph(model.to(device), dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(f"Epoch {epoch}: train: {loss} val: {val_loss}")
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))
        losses_cpu = torch.tensor(self.losses).cpu().numpy()
        val_loss_cpu = torch.tensor(self.val_loss).cpu().numpy()

        plt.figure()
        plt.plot(iters, losses_cpu, 'red', linewidth=2, label='train loss')
        plt.plot(iters, val_loss_cpu, 'coral', linewidth=2, label='val loss')

        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            smoothed_train_loss = signal.savgol_filter(losses_cpu, num, 3)
            smoothed_val_loss = signal.savgol_filter(val_loss_cpu, num, 3)

            plt.plot(iters, smoothed_train_loss, 'green', linestyle='--', linewidth=2, label='smooth train loss')
            plt.plot(iters, smoothed_val_loss, '#8B4513', linestyle='--', linewidth=2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

    def close(self):
        self.writer.close()

class ErrorRateMonitor:
    """
    错误率监视器,传入train和val的准确率,记录错误率,仅用于分类任务
    """
    def __init__(self, log_dir):
        self.save_path = os.path.join(log_dir, "Error_rate.png")
        self.fig, self.ax = plt.subplots()
        self.train_error_rates = []
        self.val_error_rates = []
        self.acc_log_path = os.path.join(log_dir, "acc_log.txt")
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Error_rate')

    def append_acc(self, epoch, train_acc, val_acc):
        train_error_rate = 1 - train_acc
        val_error_rate = 1 - val_acc

        self.train_error_rates.append(train_error_rate)
        self.val_error_rates.append(val_error_rate)

        plt.title(f'Epoch {epoch}')
        with open(self.acc_log_path, 'a') as acc_file:
            acc_file.write(
                f"Epoch {epoch}: Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}\n")
        self.error_plot(epoch)

    def error_plot(self, epoch):
        iters = np.arange(1, epoch + 1)

        self.ax.clear()
        self.ax.plot(iters, self.train_error_rates[:len(iters)], 'red', linewidth=2, label='train error')
        self.ax.plot(iters, self.val_error_rates[:len(iters)], 'coral', linewidth=2, label='val error')
        try:
            if len(self.train_error_rates) < 25:
                num = 5
            else:
                num = 15

            self.ax.plot(iters, signal.savgol_filter(self.train_error_rates[:len(iters)], num, 3), 'green',
                         linestyle='--',
                         linewidth=2, label='smooth train error')
            self.ax.plot(iters, signal.savgol_filter(self.val_error_rates[:len(iters)], num, 3), '#8B4513',
                         linestyle='--',
                         linewidth=2, label='smooth val error')
        except:
            pass

        self.ax.grid(True)
        self.ax.legend(loc="upper right")

        self.fig.savefig(self.save_path)

    def close(self):
        plt.close("all")


class ProcessMonitor:
    def __init__(self, epochs, metric='train_loss', mode='min',
                 save_path='./logs', figsize=(8, 6)):
        self.figsize = figsize
        self.save_path = os.path.join(save_path, f'{metric}.png')
        self.metric_name = metric
        self.metric_mode = mode
        self.epochs = epochs
        self.history = {}
        self.step, self.epoch = 0, 0

        self.csv_filename = os.path.join(save_path, f'{metric}.csv')
        os.makedirs(save_path, exist_ok=True)
        self._init_csv()

    def _init_csv(self):
        """初始化 CSV 文件，如果文件不存在，则创建一个包含列标题的空文件"""
        if not os.path.exists(self.csv_filename):
            columns = ['epoch', self.metric_name, 'train_loss', 'val_loss']
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.csv_filename, index=False)

    def start(self):
        print('\nView dynamic loss/metric plot: \n' + os.path.abspath(self.save_path))
        x_bounds = [0, min(10, self.epochs)]
        title = f'best {self.metric_name} = ?'
        self.update_graph(title=title, x_bounds=x_bounds)

    def log_epoch(self, info):
        self.epoch += 1
        info['epoch'] = self.epoch

        # 更新历史记录字典
        for name, metric in info.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(metric)

        # 使用 pandas 更新 CSV 文件
        self._save_to_csv()

        dfhistory = pd.DataFrame(self.history)
        n = len(dfhistory)
        x_bounds = [dfhistory['epoch'].min(), min(10 + (n // 10) * 10, self.epochs)]
        title = self.get_title()
        self.step, self.batchs = 0, self.step
        self.update_graph(title=title, x_bounds=x_bounds)

    def _save_to_csv(self):
        """将当前历史记录保存到 CSV 文件"""
        df = pd.DataFrame(self.history)
        df.to_csv(self.csv_filename, index=False)

    def end(self):
        title = self.get_title()
        self.update_graph(title=title)
        self.dfhistory = pd.DataFrame(self.history)
        return self.dfhistory

    def head(self):
        best_epoch, best_score = self.get_best_score()
        print(f"Best {self.metric_name}: {best_score:.4f} at epoch {best_epoch}")
        return self.dfhistory.head()

    def get_best_score(self):
        dfhistory = pd.DataFrame(self.history)
        arr_scores = dfhistory[self.metric_name]
        best_score = np.max(arr_scores) if self.metric_mode == "max" else np.min(arr_scores)
        best_epoch = dfhistory.loc[arr_scores == best_score, 'epoch'].tolist()[0]
        return (best_epoch, best_score)

    def get_title(self):
        best_epoch, best_score = self.get_best_score()
        title = f'best {self.metric_name}={best_score:.4f} (@epoch {best_epoch})'
        return title

    def update_graph(self, title=None, x_bounds=None, y_bounds=None):
        if not hasattr(self, 'graph_fig'):
            self.fig, self.ax = plt.subplots(1, figsize=self.figsize)
        self.ax.clear()

        dfhistory = pd.DataFrame(self.history)
        epochs = dfhistory['epoch'] if 'epoch' in dfhistory.columns else []

        metric_name = self.metric_name.replace('val_', '').replace('train_', '')

        m1 = "train_" + metric_name
        if m1 in dfhistory.columns:
            train_metrics = dfhistory[m1]
            self.ax.plot(epochs, train_metrics, 'bo--', label=m1, clip_on=False)

        m2 = 'val_' + metric_name
        if m2 in dfhistory.columns:
            val_metrics = dfhistory[m2]
            self.ax.plot(epochs, val_metrics, 'co-', label=m2, clip_on=False)

        if metric_name in dfhistory.columns:
            metric_values = dfhistory[metric_name]
            self.ax.plot(epochs, metric_values, 'co-', label=self.metric_name, clip_on=False)

        self.ax.set_xlabel("epoch")
        self.ax.set_ylabel(metric_name)

        if title:
            self.ax.set_title(title)

        if m1 in dfhistory.columns or m2 in dfhistory.columns or self.metric_name in dfhistory.columns:
            self.ax.legend(loc='best')

        if len(epochs) > 0:
            best_epoch, best_score = self.get_best_score()
            self.ax.plot(best_epoch, best_score, 'r*', markersize=15, clip_on=False)

        if x_bounds is not None: self.ax.set_xlim(*x_bounds)
        if y_bounds is not None: self.ax.set_ylim(*y_bounds)
        self.fig.savefig(self.save_path)
        plt.close()




if __name__=='__main__':
    # import random
    # epochs = 20
    # vlog = ProcessMonitor(epochs=epochs, metric='train_loss', mode='min')
    # vlog.start()
    # for epoch in range(1, epochs + 1):
    #     train_loss = random.uniform(0.5, 2.0)
    #     val_loss = random.uniform(0.4, 1.5)
    #     vlog.log_epoch({'train_loss': train_loss, 'val_loss': val_loss})
    #
    # vlog.end()
    # print(vlog.head())
    #
    #
    # vlog = ProcessMonitor(epochs=epochs, metric='psnr', mode='max')
    # for epoch in range(1, epochs + 1):
    #     psnr = random.uniform(25, 42)
    #     vlog.log_epoch({'psnr': psnr})
    #
    # vlog.end()
    # print(vlog.head())

    true = torch.tensor([0, 1, 2, 0, 3, 2, 4])
    pred = torch.tensor([0, 1, 1, 0, 3, 1, 4])  # 第三个样本预测错误
    cm = EvalMetrics(num_classes=5)
    cm.update(true, pred)
    cm.eval()






