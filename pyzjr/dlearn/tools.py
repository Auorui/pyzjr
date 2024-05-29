import time
import torch
import argparse
import torch.nn as nn
from pyzjr.visualize.video_utils import Timer
from pyzjr.visualize.printf import colorstr
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

__all__ = ["Runcodes", "LoadingBar", "show_config", "time_sync", "profile"]

class Runcodes:
    """
    Comparing the running time of different algorithms.
    example:
        with Runcodes("inference time"):
            output = ...
    """
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.5f} sec')


class LoadingBar:
    def __init__(self, length: int = 40):
        """Simple dynamic display bar
        example:
            loading_bar = LoadingBar()
            for i in range(101):
                progress = i / 100.0
                bar_string = loading_bar(progress)
                print(f"\r[{bar_string}] {i}% ", end="")
                time.sleep(0.1)  # 模拟加载延迟
            print("\nLoading Complete!")
        """
        self.length = length
        self.symbols = ['┈', '░', '▒', '▓']

    def __call__(self, progress: float) -> str:
        p = int(progress * self.length*4 + 0.5)
        d, r = p // 4, p % 4
        return '┠┈' + d * '█' + ((self.symbols[r]) + max(0, self.length-1-d) * '┈' if p < self.length*4 else '') + "┈┨"

def show_config(head="Configurations", args=None, color='black', **kwargs):
    """显示配置信息"""
    print(colorstr(color, f'{head}:'))
    print(colorstr(color, '-' * 113))
    print(colorstr(color, '|%5s | %45s | %55s|' % ('order', 'keys', 'values')))
    print(colorstr(color, '-' * 113))
    counter = 0
    if args is not None:
        if isinstance(args, argparse.Namespace):
            config_dict = vars(args)
            for key, value in config_dict.items():
                counter += 1
                print(colorstr(color, f'|%5d | %45s | %55s|' % (counter, key, value)))
        elif isinstance(args, list):
            for arg in args:
                counter += 1
                print(colorstr(color, f'|%5d | %45s | {"": >55}|' % (counter, arg)))  # Assuming each element in the list is a key
        else:
            counter += 1
            print(colorstr(color, f'|%5d | %45s | %55s|' % (counter, "arg", args)))  # Assuming args is a single value

    for key, value in kwargs.items():
        counter += 1
        print(colorstr(color, f'|%5d | %45s | %55s|' % (counter, key, value)))

    print(colorstr(color, '-' * 113))

def profile(input, ops, n=10, cuda=True):
    """
    Usage:
        model = MyModel().to('cuda')
        input_tensor = torch.randn(1, 3, 512, 512).to('cuda')
        profile(input_tensor, model, n=10)
    """
    results = []
    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s}")
    if torch.cuda.is_available():
        device = "cuda" if cuda else "cpu"
    else:
        device = "cpu"
    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, 'to') else m
            m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = thop.profile(m, inputs=(x, ), verbose=False)[0] / 1E9 * 2  # GFLOPs
            except Exception:
                flops = 0
            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception as e:  # no backward method
                        print(e)
                        t[2] = float('nan')
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else 'list' for x in (x, y))  # shapes
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0  # parameters
                print(f'{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}')
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results

def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

if __name__=="__main__":
    import platform
    import sys

    show_config(Python_vision=".".join(map(str, [sys.version_info.major, sys.version_info.minor, sys.version_info.micro])),
                CUDA=torch.version.cuda,
                CUDA_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                Torch_vision=torch.__version__,
                os_info=platform.system() + " " + platform.version(),
                color='red',
                )

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
            self.fc = nn.Linear(128 * 512 * 512, 10)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = MyModel().to('cpu')
    input_tensor = torch.randn(1, 3, 512, 512).to('cpu')
    results = profile(input_tensor, model, n=10)