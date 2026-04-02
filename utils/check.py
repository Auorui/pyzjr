"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is used to check the input type.
"""
import re
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import platform
from typing import Iterable

def is_tensor(param):
    """Check if the input image is torch format."""
    return isinstance(param, torch.Tensor)

def is_pil(param):
    """Check if the input image is PIL format."""
    return isinstance(param, Image.Image)

def is_numpy(param):
    """Check if the input image is Numpy format."""
    return isinstance(param, np.ndarray)

def is_gray_image(param):
    return (len(param.shape) == 2) or (len(param.shape) == 3 and param.shape[-1] == 1)

def is_rgb_image(param):
    return len(param.shape) == 3 and param.shape[-1] == 3

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))

def is_Iterable(param):
    return isinstance(param, Iterable)

def is_str(param):
    return isinstance(param, str)

def is_int(param):
    return isinstance(param, int)

def is_float(param):
    return isinstance(param, float)

def is_bool(param):
    return param.dtype == np.bool_

def is_list(param):
    return isinstance(param, list)

def is_tuple(param):
    return isinstance(param, tuple)

def is_list_or_tuple(param):
    return isinstance(param, (list, tuple))

def is_none(param):
    return True if param is None else False

def is_not_none(param):
    return not is_none(param)

def is_positive_int(param):
    return is_int(param) and param > 0

def is_nonnegative_int(param):
    return is_int(param) and param >= 0

def is_ascii(param):
    """Check if the string is composed of only ASCII characters."""
    s = str(param)
    return all(ord(c) < 128 for c in s)

def is_url(param):
    """Return True if string is an http or ftp path."""
    URL_REGEX = re.compile(r'http://|https://|ftp://|file://|file:\\')
    return (isinstance(param, str) and
            URL_REGEX.match(param) is not None)

def is_image_extension(param):
    ext = param.split('.')
    return ext[-1] in ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm') and is_str(image_name)

def is_video_extension(param):
    ext = param.split('.')
    return ext[-1] in ('asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv') and is_str(video_name)

def is_file(param):
    return os.path.isfile(param) and is_str(param)

def is_directory(param):
    return os.path.isdir(param) and is_str(param)

def is_directory_not_empty(param):
    return os.path.isdir(param) and len(os.listdir(param)) > 0 and is_str(param)

def is_path_exists(param):
    return os.path.exists(param) and is_str(param)

def is_windows():
    if platform.system() == "Windows":
        return True

def is_linux():
    if platform.system() == "Linux":
        return True

def is_odd(param):
    return param % 2 != 0

def is_even(param):
    return param % 2 == 0

def is_list_of_tensor(param) -> bool:
    if not isinstance(param, list):
        return False
    if len(param) == 0:
        return True
    for item in param:
        if not isinstance(item, torch.Tensor):
            return False

    return True

def is_list_of_numpy(param) -> bool:
    if not isinstance(param, list):
        return False
    if len(param) == 0:
        return True
    for item in param:
        if not isinstance(item, np.ndarray):
            return False
    return True


if __name__=="__main__":
    image_name = r'D:\PythonProject\pyzjrPyPi\pyzjr\augmentation\test.png'
    print(is_file(image_name), is_windows())