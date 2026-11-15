"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is used for outputting console information.
"""
import os
import re
import sys
import yaml
import time
import math
import pprint
import argparse
import numpy as np
from datetime import datetime


def format_num(x, mode="round", decimals=4):
    """
    Format a number or a numeric string into a specific format.

    Args:
        x: Input value (int, float, or numeric string).
        mode (str): The formatting mode. Options:
            - "round": Standard rounding (四舍五入) to `decimals` places.
            - "ceil": Round up to the nearest integer (向上取整).
            - "floor": Round down to the nearest integer (向下取整).
        decimals (int): Number of decimal places for "round" mode. Defaults to 4.

    Returns:
        Formatted number (or the original input if it cannot be converted).
    """
    try:
        val = float(x)
    except (ValueError, TypeError):
        return x

    try:
        if mode == "ceil":
            return math.ceil(val)
        elif mode == "floor":
            return math.floor(val)
        else:  # The default is 'round'
            return round(val, decimals)
    except Exception:
        return x


class Timer:
    def __init__(self):
        """Start is called upon creation"""
        self.times = []
        self.start()

    def start(self):
        """initial time"""
        self.gap = time.time()

    def stop(self):
        """The time interval from the start of timing to calling the stop method"""
        self.times.append(time.time() - self.gap)
        return self.times[-1]

    def avg(self):
        """Average time consumption"""
        return sum(self.times) / len(self.times)

    def total(self):
        """Total time consumption"""
        return sum(self.times)

    def cumsum(self):
        """Accumulated sum of time from the previous n runs"""
        return np.array(self.times).cumsum().tolist()


class Runcodes:
    """
    Comparing the running time of different algorithms.
    Examples:
    ```
        with Runcodes("inference time"):
            output = ...
    ```
    """
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.7f} sec')


def timestr(use_simple_format=True):
    """Generate a formatted datetime string."""
    fmt = "%Y_%m_%d_%H_%M_%S" if use_simple_format else "%A_%d_%B_%Y_%Hh_%Mm_%Ss"
    return f"{datetime.now().strftime(fmt)}"


def colorstr(*args, string=None, color="red", display_type="plain"):
    """
    Add ANSI escape codes to a string for colored terminal output.

    Usage Examples:
        >>> colorstr('blue', 'bold', 'Hello world!')
        >>> colorstr('Hello world!', color='red', display_type='underline')
        >>> colorstr('red', 'Hello world!')
        >>> colorstr('Hello world!')

    Args:
        *args: Flexible positional arguments. Can contain style keywords (e.g., 'blue', 'bold')
               and the target string itself.
        string (str, optional): The target string. If not provided, it will be inferred from `args`.
        color (str, optional): Foreground color name (e.g., 'red', 'green'). Overrides positional colors.
        display_type (str, optional): Display type (e.g., 'plain', 'highlight', 'underline').
                # 0: Reset all attributes to default
                # 1: Bold or increased intensity
                # 4: Underline
                # 5: Slow blink
                # 7: Reverse video or invert
                # 8: Conceal or hide
    Returns:
        str: The formatted string with ANSI escape codes.
    """
    colors = {
        'black': '\033[30m', 'red': '\033[31m', 'green': '\033[32m',
        'yellow': '\033[33m', 'blue': '\033[34m', 'magenta': '\033[35m', 'purple': '\033[35m',
        'cyan': '\033[36m', 'white': '\033[37m',
        # Bright color
        'bright_black': '\033[90m', 'bright_red': '\033[91m', 'bright_green': '\033[92m',
        'bright_yellow': '\033[93m', 'bright_blue': '\033[94m', 'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m', 'bright_white': '\033[97m',
        # Style / Display Type
        'plain': '\033[0m', 'bold': '\033[1m', 'highlight': '\033[1m', 'underline': '\033[4m',
        'shine': '\033[5m', 'inverse': '\033[7m', 'invisible': '\033[8m', 'end': '\033[0m'
    }

    parsed_args = []
    if args:
        if len(args) == 1:
            string = string or args[0]
        else:
            for arg in args:
                if arg in colors:
                    parsed_args.append(colors[arg])
                else:
                    string = string or arg

    if display_type and display_type in colors:
        parsed_args.append(colors[display_type])
    if color and color in colors:
        parsed_args.append(colors[color])

    string = string if string is not None else ""
    return ''.join(parsed_args) + str(string) + colors['end']


def printc(*args, color="cyan", display_type="plain", indent=2, width=80, sep=' ', end='\n'):
    """
    A hybrid of print() and pprint() with ANSI color support.

    - Complex objects (dict, list, tuple, set) are pretty-printed.
    - Multiple arguments are joined like normal print().

    Args:
        *args: The objects to print.
        color (str): The text color. Defaults to "cyan".
        display_type (str): Display type (e.g., 'plain', 'highlight', 'underline').
        indent (int): Indent for pretty-printing. Defaults to 2.
        width (int): Max width for pretty-printing. Defaults to 80.
        sep (str): Separator between arguments. Defaults to ' '.
        end (str): String appended after the last value. Defaults to '\\n'.
    """
    formatted_args = []
    for arg in args:
        if isinstance(arg, (dict, list, tuple, set)):
            formatted_args.append(
                pprint.pformat(arg, indent=indent, width=width, compact=False, sort_dicts=False)
            )
        else:
            formatted_args.append(str(arg))

    final_str = sep.join(formatted_args)
    print(colorstr(final_str, color=color, display_type=display_type), end=end)


def flatten_dict(d, parent_key='', sep='.'):
    """Flatten nested dictionaries into dot-separated key-value pairs"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                list_key = f"{new_key}[{i}]"
                if isinstance(item, (dict, list)):
                    items.extend(flatten_dict({list_key: item}).items())
                else:
                    items.append((list_key, item))
        else:
            items.append((new_key, v))
    return dict(items)


def show_config(head="Configurations", args=None, yaml_path=None, color='black', **kwargs):
    """
    Display configuration information with automatic line-wrapping for long strings.
    Supports command-line arguments, YAML files, and keyword arguments.

    Args:
        head (str): The title header of the configuration table. Defaults to "Configurations".
        args (argparse.Namespace or list, optional): Command-line arguments to display.
        yaml_path (str, optional): Path to the YAML configuration file to load and display.
        color (str): The console output color (e.g., 'black', 'red', 'green'). Defaults to 'black'.
        **kwargs: Additional key-value pairs to include in the configuration table.

    Returns:
        dict: A flattened dictionary containing all the merged configurations.
    """
    print(colorstr(color, f'{head} - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'))
    print(colorstr(color, '-' * 113))
    print(colorstr(color, '|%5s | %45s | %55s|' % ('order', 'keys', 'values')))
    print(colorstr(color, '-' * 113))

    counter = 0
    config_dict = {}

    if args is not None:
        if isinstance(args, argparse.Namespace):
            cmd_config = vars(args)
            config_dict.update(cmd_config)
        elif isinstance(args, list):
            for i, arg in enumerate(args):
                config_dict[f'arg_{i}'] = arg
        else:
            config_dict['arg'] = args

    if yaml_path is not None:
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            flattened_yaml = flatten_dict(yaml_config)
            config_dict.update(flattened_yaml)
        except Exception as e:
            print(colorstr('red', f"Error loading YAML file {yaml_path}: {e}"))

    config_dict.update(kwargs)

    MAX_LEN = 55

    for key, value in config_dict.items():
        counter += 1
        str_value = str(value)
        if len(str_value) <= MAX_LEN:
            print(colorstr(color, f'|%5d | %45s | %55s|' % (counter, key, str_value)))
        else:
            chunks = [str_value[i:i + MAX_LEN] for i in range(0, len(str_value), MAX_LEN)]
            print(colorstr(color, f'|%5d | %45s | %55s|' % (counter, key, chunks[0])))
            for chunk in chunks[1:]:
                print(colorstr(color, f'|%5s | %45s | %55s|' % ('', '', chunk)))

    print(colorstr(color, '-' * 113))

    return config_dict


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
        # To be compatible with some environments that do not support flush
        if hasattr(self.terminal, 'flush'):
            self.terminal.flush()

    @staticmethod
    def remove_ansi_colors(text):
        """Remove ANSI color codes."""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)


def redirect_console(log_path='./out.log'):
    """
    Redirect console output to a specified file.

    Args:
        log_path (str): The path of the log file
    """
    logger = ConsoleLogger(log_path)
    sys.stdout = logger  # Redirect standard output to Logger



if __name__=="__main__":
    print(timestr(), timestr(False))

    printc(format_num(str(3.1415926), decimals=4))

    printc({"Epoch": 10, "Loss": 0.05, "Data": {"path": "/very/long/path/to/dataset", "shape": [512, 512]}}, color="yellow")

    show_config(Python_vision=".".join(map(str, [sys.version_info.major, sys.version_info.minor, sys.version_info.micro])),
                image_dir=r'E:/PythonProject/SegLab/data/Concrete400/very/long/path/exceeds/the/limit',
                lr=2e-4,
                color='red',)

    colors = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'purple', 'cyan', 'white',
              'bright_black', 'bright_red', 'bright_green', 'bright_yellow', 'bright_blue',
              'bright_magenta', 'bright_cyan', 'bright_white']

    display_types = ["plain", "highlight", "underline", "shine", "inverse", "invisible"]

    text = "Hello, Colorful World!"

    # for color in colors:
    #     for display_type in display_types:
    #         colored_text = colorstr(text, color=color, display_type=display_type)
    #         print(f"Color: {color}, Display Type: {display_type} -> {colored_text}")

