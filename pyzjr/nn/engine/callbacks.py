import os
import torch
from collections import deque
import matplotlib
matplotlib.use('Agg')
import scipy.signal as signal
import matplotlib.pyplot as plt
from pyzjr.data.core.path import natsorted

__all__ = ["AverageMeter", "SmoothedValue", "DataStatsCalculator", "LossHistory"]

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

class DataStatsCalculator:
    def __init__(self):
        self.data = {}
        self.total = 0.0
        self.count = 0
        self.max_val = -float('inf')
        self.min_val = float('inf')
        self.max_ids = []
        self.min_ids = []

    def update(self, identifier, value, verbose=False):
        """Update statistics with new data point"""
        self.data[identifier] = value
        self.total += value
        self.count += 1

        # Update max value and identifiers
        if value > self.max_val:
            self.max_val = value
            self.max_ids = [identifier]
        elif value == self.max_val:
            self.max_ids.append(identifier)

        # Update min value and identifiers
        if value < self.min_val:
            self.min_val = value
            self.min_ids = [identifier]
        elif value == self.min_val:
            self.min_ids.append(identifier)

        if verbose:
            print(f"Added: {identifier} = {value}")

    def calculate(self):
        """Calculate and return statistics"""
        if self.count == 0:
            return None

        avg = self.total / self.count
        return {
            'average': avg,
            'max_value': self.max_val,
            'max_identifiers': self.max_ids,
            'min_value': self.min_val,
            'min_identifiers': self.min_ids,
            'count': self.count
        }

    def save(self, filename='statistics.txt'):
        """Save results to file"""
        stats = self.calculate()
        if stats is None:
            print("No valid data to save!")
            return
        sorted_keys = natsorted(self.data.keys())
        with open(filename, 'w') as f:
            f.write("ID\tValue\n")
            for key in sorted_keys:
                f.write(f"{key}\t{self.data[key]:.4f}\n")

            f.write("\nStatistics:\n")
            f.write(f"Average: {stats['average']:.4f}\n")
            f.write(f"Max: {stats['max_value']:.4f} (IDs: {', '.join(stats['max_identifiers'])})\n")
            f.write(f"Min: {stats['min_value']:.4f} (IDs: {', '.join(stats['min_identifiers'])})\n")
            f.write(f"Total entries: {stats['count']}\n")


class LossHistory:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []
        os.makedirs(self.log_dir, exist_ok=True)
        self.txt_path = os.path.join(self.log_dir, "epoch_loss.txt")

    def append_loss(self, epoch, loss, val_loss, plot_freq=10):
        """
        Record data and automatically generate charts according to the set frequency.

        Args:
            epoch (int): The current round (starting from 1).
            loss (float): Training Set Loss
            val_loss (float): Validation Set Loss
            plot_freq (int): Plotting frequency. How many rounds to plot and save a graph. Default: 10.
        """
        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(self.txt_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch} train {loss:.5f} val {val_loss:.5f}\n")

        if epoch % plot_freq == 0 or epoch == 1:
            self.plot_loss()

    def plot_loss(self):
        """Plot and save the current loss curve"""
        if len(self.losses) == 0:
            return

        iters = range(1, len(self.losses) + 1)
        losses_cpu = torch.tensor(self.losses).cpu().numpy()
        val_loss_cpu = torch.tensor(self.val_loss).cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.plot(iters, losses_cpu, 'red', linewidth=2, label='train loss')
        plt.plot(iters, val_loss_cpu, 'coral', linewidth=2, label='val loss')

        if len(self.losses) >= 5:
            try:
                num = min(15, len(self.losses))
                if num % 2 == 0:
                    num -= 1
                if num > 3:
                    smoothed_train_loss = signal.savgol_filter(losses_cpu, num, 3)
                    smoothed_val_loss = signal.savgol_filter(val_loss_cpu, num, 3)
                    plt.plot(iters, smoothed_train_loss, 'green', linestyle='--', linewidth=2, label='smooth train loss')
                    plt.plot(iters, smoothed_val_loss, '#8B4513', linestyle='--', linewidth=2, label='smooth val loss')
            except Exception:
                pass

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        save_path = os.path.join(self.log_dir, "epoch_loss.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.cla()
        plt.close("all")

    def finalize(self):
        """
        Finalize the training process by drawing and saving the final loss curve.
        Call this method at the end of the training loop to ensure the last epoch is plotted.
        """
        self.plot_loss()

if __name__ == "__main__":
    import random

    print("=" * 50)
    print("测试 AverageMeter (基础平均值追踪)")
    print("=" * 50)
    meter = AverageMeter()
    # 模拟 5 个 batch 的 loss，最后一个 batch 只有 2 个样本（模拟 drop_last=False）
    mock_losses = [2.0, 4.0, 3.0, 1.0, 0.5]
    mock_bs = [32, 32, 32, 32, 2]
    for loss, bs in zip(mock_losses, mock_bs):
        meter.update(loss, n=bs)
        print(f"当前 Batch Loss: {meter.val:.4f} | 累计平均 Loss: {meter.avg:.4f}")

    print("\n" + "=" * 50)

    print("测试 DataStatsCalculator (带 ID 的最值统计)")
    print("=" * 50)
    stats_calc = DataStatsCalculator()
    # 模拟统计每张图的目标数量
    mock_data = {
        "image_001.jpg": 3.0,
        "image_002.jpg": 8.0,
        "image_010.jpg": 8.0, # 并列最大
        "image_003.jpg": 1.0,
        "image_004.jpg": 5.0,
    }
    for k, v in mock_data.items():
        stats_calc.update(k, v)

    result = stats_calc.calculate()
    print("计算返回结果:", result)

    # 测试保存到文件
    # test_txt = "test_stats.txt"
    # stats_calc.save(test_txt)
    # print(f"\n成功保存至 {test_txt}，内容如下：")
    # with open(test_txt, "r", encoding="utf-8") as f:
    #     print(f.read())
    # os.remove(test_txt) # 清理测试文件

    print("\n" + "=" * 50)
    print("测试 SmoothedValue (平滑窗口追踪)")
    print("=" * 50)
    smoothed = SmoothedValue(window_size=5)
    for i, loss in enumerate([1.2, 1.1, 1.3, 1.05, 0.9, 0.8, 0.85, 0.7]):
        smoothed.update(loss)
        # 这里的 str(smoothed) 就是用来在 tqdm 里打印的格式
        print(f"Step {i+1}: {str(smoothed)}")

    print("\n" + "=" * 50)
    print("4. 测试 LossHistory (真实曲线绘制与平滑)")
    print("=" * 50)

    test_log_dir = "./test_loss_history"
    history = LossHistory(log_dir=test_log_dir)

    for epoch in range(1, 51):
        train_loss = 2.0 / (epoch ** 0.5) + random.uniform(-0.05, 0.05)
        # 伪造 val loss (下降更慢，且有明显波动)
        val_loss = 1.8 / (epoch ** 0.4) + random.uniform(-0.15, 0.15)

        history.append_loss(epoch, train_loss, val_loss)
        if epoch % 10 == 0:
            print(f"已记录并绘制到 Epoch: {epoch}")

    print(f"\n绘图完成！请去 {test_log_dir} 目录下查看 epoch_loss.png 文件。")