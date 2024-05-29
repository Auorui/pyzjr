import os
import numpy as np
import pandas as pd
import matplotlib
import torch
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal as signal
from torch.utils.tensorboard import SummaryWriter

__all__ = ["LossHistory", "ErrorRateMonitor", "LossMonitor"]

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []

        os.makedirs(self.log_dir,exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1]).to(device)
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(f"Epoch {epoch}: Train: {loss}, Validation: {val_loss}")
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
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
    错误率监视器,传入train和val的准确率,记录错误率
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


class LossMonitor:
    def __init__(self, epochs, metric='train_loss', mode='min',
                 save_path='history_loss.png', figsize=(8, 6)):
        self.figsize = figsize
        self.save_path = save_path
        self.metric_name = metric
        self.metric_mode = mode
        self.epochs = epochs
        self.history = {}
        self.step, self.epoch = 0, 0

    def start(self):
        print('\nView dynamic loss/metric plot: \n' + os.path.abspath(self.save_path))
        x_bounds = [0, min(10, self.epochs)]
        title = f'best {self.metric_name} = ?'
        self.update_graph(title=title, x_bounds=x_bounds)

    def log_epoch(self, info):
        self.epoch += 1
        info['epoch'] = self.epoch
        for name, metric in info.items():
            self.history[name] = self.history.get(name, []) + [metric]
        dfhistory = pd.DataFrame(self.history)
        n = len(dfhistory)
        x_bounds = [dfhistory['epoch'].min(), min(10 + (n // 10) * 10, self.epochs)]
        title = self.get_title()
        self.step, self.batchs = 0, self.step
        self.update_graph(title=title, x_bounds=x_bounds)

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
    import random
    epochs = 20
    vlog = LossMonitor(epochs=epochs, metric='train_loss', mode='min')

    for epoch in range(1, epochs + 1):
        train_loss = random.uniform(0.5, 2.0)
        val_loss = random.uniform(0.4, 1.5)
        vlog.log_epoch({'train_loss': train_loss, 'val_loss': val_loss})

    df_history = vlog.end()
    print(vlog.head())
