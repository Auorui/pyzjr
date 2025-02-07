import re
import os
from pathlib import Path
from datetime import datetime

def timestr(use_simple_format=True):
    """Generate a formatted datetime string."""
    format = "%Y_%m_%d_%H_%M_%S" if use_simple_format else "%A_%d_%B_%Y_%Hh_%Mm_%Ss"
    return f"{datetime.now().strftime(format)}"

def alphabetstr(capital=False):
    """字符标签"""
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    uppercase_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    return uppercase_alphabet if capital else alphabet

def natural_key(st):
    """
    将字符串拆分成字母和数字块
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', st)]

def natsorted(a):
    """
    手写自然排序
    >>> a = ['num9', 'num5', 'num2']
    >>> sorted_a = natsorted(a)
    ['num2', 'num5', 'num9']
    """
    return sorted(a, key=natural_key)

def list_dirs(dirs):
    """
    返回给定目录中的所有目录
    """
    currentdirs = [str(f) for f in Path(dirs).iterdir() if f.is_dir()]
    return natsorted(currentdirs), len(currentdirs)

def list_files(dirs):
    """
    返回给定目录中的所有文件
    """
    currentfiles = [
        str(f) for f in Path(dirs).iterdir()
        if f.is_file() and not f.name.startswith(".")
    ]
    return natsorted(currentfiles), len(currentfiles)

def list_names(file_dir):
    """获取指定文件夹下的文件列表，并按自然排序方式排序."""
    files = natsorted(os.listdir(file_dir))
    files_num = len(files)
    return files, files_num

if __name__=="__main__":
    paths = r'D:\PythonProject\pyzjrPyPi\pyzjr'
    dirs, nums = list_dirs(paths)
    print(dirs, nums)
    files, nums = list_files(paths)
    print(files, nums)
    files, nums = list_names(paths)
    print(files, nums)

