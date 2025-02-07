"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is used for outputting console information.
"""
import os
import re
import sys
import datetime
import argparse

def printlog(info):
    """Print log information with timestamp."""
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s" % nowtime)
    print(info+'...\n\n')

def printprocess(info, index, nums):
    """打印进度信息，显示当前处理的进度"""
    print(f"\r[{info}] processing [{index}/{nums}], rate:{index/nums*100:.1f}%", end="")

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

class LoadingBar:
    def __init__(self, length: int = 40):
        """Simple dynamic display bar
        Example:
        ```
            loading_bar = LoadingBar()
            for i in range(101):
                progress = i / 100.0
                bar_string = loading_bar(progress)
                print(f"\r[{bar_string}] {i}% ", end="")
                time.sleep(0.1)
            print("\nLoading Complete!")
        ```
        """
        self.length = length
        self.symbols = ['┈', '░', '▒', '▓']

    def __call__(self, progress: float) -> str:
        p = int(progress * self.length*4 + 0.5)
        d, r = p // 4, p % 4
        return '┠┈' + d * '█' + ((self.symbols[r]) + max(0, self.length-1-d) * '┈' if p < self.length*4 else '') + "┈┨"


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
    log_file = './output.log'

    redirect_console(log_file)
    print("This is a test message.")
    print("Another message to log.")

    total_tasks = 15
    task_info = "Task A"
    for i in range(1, total_tasks + 1):
        printprocess(task_info, i, total_tasks)
    print("\n任务完成！")

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