"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is used for outputting console information.
"""
import os
import re
import sys
import time
import argparse
from datetime import datetime

def printshape(*args):
    for arg in args:
        try:
            shape = arg.shape
            print(f"{arg.__class__.__name__}: {shape}")
        except AttributeError:
            print(f"Error: The object {arg} does not have a shape attribute.")

def printlog(info):
    """Print log information with timestamp."""
    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s" % nowtime)
    print(info+'...\n\n')

def printcolor(*args, color="black", display_type="plain", sep=' ', end='\n', flush=False):
    """
    打印着色并设置显示方式的字符串。
    color: "black", "red", "green", "yellow", "blue", "purple", "cyan", "white"
    display_type: "plain", "highlight", "underline", "shine", "inverse", "invisible"
    # 显示方式           意义
    # -------------------------
    # 0                终端默认设置
    # 1                高亮显示
    # 4                使用下划线
    # 5                闪烁
    # 7                反白显示
    # 8                不可见
    """
    colored_strs = [colorfulstr(arg, color, display_type) for arg in args]
    print(*colored_strs, sep=sep, end=end, flush=flush)

def printprocess(info, index, nums, color="black", display_type="plain"):
    """打印进度信息，显示当前处理的进度"""
    printcolor(f"\r[{info}] processing [{index}/{nums}], rate:{index/nums*100:.1f}%",
               color=color, display_type=display_type, end="", flush=True)

def colorfulstr(obj, color="red", display_type="plain"):
    """
    Advanced version of "colorstr" function
    Args:
        obj: info content
        color: "black", "red", "green", "yellow", "blue", "purple", "cyan", "white"
        display_type: "plain", "highlight", "underline", "shine", "inverse", "invisible"

    Details:
        # 彩色输出格式：
        # 设置颜色开始 ：\033[显示方式;前景色;背景色m
        # 前景色            背景色            颜色
        # ---------------------------------------
        # 30                40              黑色
        # 31                41              红色
        # 32                42              绿色
        # 33                43              黃色
        # 34                44              蓝色
        # 35                45              紫红色
        # 36                46              青蓝色
        # 37                47              白色
        # 显示方式           意义
        # -------------------------
        # 0                终端默认设置
        # 1                高亮显示
        # 4                使用下划线
        # 5                闪烁
        # 7                反白显示
        # 8                不可见
    """
    color_dict = {"black": "30", "red": "31", "green": "32", "yellow": "33",
                  "blue": "34", "purple": "35","cyan": "36",  "white": "37"}
    display_type_dict = {"plain": "0", "highlight": "1", "underline": "4",
                         "shine": "5", "inverse": "7", "invisible": "8"}
    s = str(obj)
    color_code = color_dict.get(color, "")
    display = display_type_dict.get(display_type, "")
    out = '\033[{};{}m'.format(display, color_code) + s + '\033[0m'
    return out

def colorstr(*input):
    # https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def show_config(head="Configurations", args=None, color='black', **kwargs):
    """显示配置信息"""
    print(colorstr(color, f'{head} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'))
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

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1,
                     length=50, fill : str or int='█'):
    """
    Display an animated progress bar in the console with time estimation
    :param iteration: (int) Current iteration index (0-based)
    :param total: (int) Total number of iterations
    :param prefix: (str) Text shown before the progress bar
    :param suffix: (str) Text shown after the progress bar
    :param decimals: (int) Number of decimal places in percentage (0-6)
    :param length: (int) Character length of the progress bar (10-200)
    :param fill: (int/str) Fill character for progress visualization:
                - int: Index for predefined fill characters ['█', '▓', '▒', '░', ...]
                - str: First character of the string will be used
    """
    fill_icon = [
        # 经典方块系列
        '█', '▓', '▒', '░',
        # 渐变块状系列
        '🮇', '🮈', '🮉', '🮊',
        '▏', '▎', '▍', '▌',
        # 箭头系列
        '➔', '➤', '➪', '⏩', '▶', '▷', '⯈',
        # 几何形状
        '◆', '◇', '◈',
        '●', '○', '◎',
        '■', '□', '◼',
        # 特殊符号
        '★', '☆', '☀', '❤', '#'
        '⚡', '🌀', '🎯', '🎲', '🚀'
        # 创意组合
        '⣿', '⣾', '⣽', '⣼', '⣻', '⣺',  # 组合块
        '⠁', '⠂', '⠄', '⡀', '⢀', '⡠'  # 点阵组合
    ]
    if isinstance(fill, int):
        selected_fill = fill_icon[fill % len(fill_icon)]
    else:
        selected_fill = str(fill)[0] if len(str(fill)) > 0 else '█'

    if not hasattr(printProgressBar, "start_time"):
        printProgressBar.start_time = time.time()

    elapsed_time = time.time() - printProgressBar.start_time
    elapsed_str = f"{int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}"

    if iteration > 0:
        remaining = elapsed_time * (total - iteration) / iteration
        remaining_str = f"{int(remaining // 60):02d}:{int(remaining % 60):02d}"
    else:
        remaining_str = "??:??"

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = selected_fill * filled_length + '-' * (length - filled_length)

    time_info = f"{elapsed_str}<{remaining_str}"
    suffix = f"[{time_info}, {suffix}]"

    sys.stdout.write('\r\033[K')
    sys.stdout.write(f'{prefix} {percent}% |{bar}| {iteration}/{total} {suffix}')
    sys.stdout.flush()

    if iteration == total:
        del printProgressBar.start_time
        sys.stdout.write('\n')


class ConsoleLogger:
    def __init__(self, log_file, encoding='utf-8'):
        self.log_file = log_file
        self.clean(self.log_file)
        self.terminal = sys.stdout
        self.encoding = encoding

    def write(self, message):
        message_no_color = self.remove_ansi_colors(message)
        with open(self.log_file, 'a', encoding=self.encoding) as log:
            log.write(message_no_color)

        self.terminal.write(message)

    def clean(self,log_file):
        if os.path.exists(log_file):
            os.remove(log_file)

    def flush(self):
        # 为了兼容一些不支持flush的环境
        self.terminal.flush()

    @staticmethod
    def remove_ansi_colors(text):
        """
        去除 ANSI 颜色代码。
        """
        # 正则表达式匹配 ANSI 转义序列
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)


def redirect_console(log_path='./out.log'):
    """
    将控制台输出重定向到指定文件。

    Args:
        log_path (str): 日志文件的路径。
    """
    logger = ConsoleLogger(log_path)
    sys.stdout = logger  # 将标准输出重定向到 Logger

if __name__ == "__main__":
    import platform
    import torch
    # 设置日志文件路径
    # log_file = './output.log'
    #
    # redirect_console(log_file)
    print("This is a test message.")
    print("Another message to log.")

    total_tasks = 15
    task_info = "Task A"
    for i in range(1, total_tasks + 1):
        printprocess(task_info, i, total_tasks)
        time.sleep(1)
    print("\n任务完成！")

    tasks = 100
    import random
    for i in range(tasks):
        time.sleep(0.2)
        printProgressBar(
            iteration=i + 1,
            total=tasks,
            prefix=f'Epoch{i+1}/{tasks}: ',
            suffix=f'psnr:{random.randint(10, 20)}, ssim:{random.random():.2f}',
            fill=12,
            decimals=2,
            length=30,
        )

    text = "Hello, Colorful World!"
    printlog("Starting the main process.")

    colors = ["red", "green", "blue", "yellow", "purple", "cyan", "white"]
    display_types = ["plain", "highlight", "underline", "shine", "inverse", "invisible"]

    show_config(Python_vision=".".join(map(str, [sys.version_info.major, sys.version_info.minor, sys.version_info.micro])),
                CUDA=torch.version.cuda,
                CUDA_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                Torch_vision=torch.__version__,
                os_info=platform.system() + " " + platform.version(),
                color='red',)

    for color in colors:
        for display_type in display_types:
            colored_text = colorfulstr(text, color=color, display_type=display_type)
            print(f"Color: {color}, Display Type: {display_type} -> {colored_text}")

    printcolor("Hello", "world", color="red")
    printcolor("First line", color="blue", end=" - ")
    printcolor("Second line")
    for i in range(5):
        printcolor(f"\rLoading {i+1}/5", color="blue", end='', flush=True)
        time.sleep(1)
    print("\nDone!")
