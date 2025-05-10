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

def is_tensor(img):
    """Check if the input image is torch format."""
    return isinstance(img, torch.Tensor)

def is_pil(img):
    """Check if the input image is PIL format."""
    return isinstance(img, Image.Image)

def is_numpy(img):
    """Check if the input image is Numpy format."""
    return isinstance(img, np.ndarray)

def is_gray_image(image):
    return (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1)

def is_rgb_image(image):
    return len(image.shape) == 3 and image.shape[-1] == 3

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

def is_ascii(s):
    """Check if the string is composed of only ASCII characters."""
    s = str(s)
    return all(ord(c) < 128 for c in s)

def is_url(filename):
    """Return True if string is an http or ftp path."""
    URL_REGEX = re.compile(r'http://|https://|ftp://|file://|file:\\')
    return (isinstance(filename, str) and
            URL_REGEX.match(filename) is not None)

def is_image_extension(image_name):
    ext = image_name.split('.')
    return ext[-1] in ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm') and is_str(image_name)

def is_video_extension(video_name):
    ext = video_name.split('.')
    return ext[-1] in ('asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv') and is_str(video_name)

def is_file(path):
    return os.path.isfile(path) and is_str(path)

def is_directory(path):
    return os.path.isdir(path) and is_str(path)

def is_directory_not_empty(path):
    return os.path.isdir(path) and len(os.listdir(path)) > 0 and is_str(path)

def is_path_exists(path):
    return os.path.exists(path) and is_str(path)

def is_windows():
    if platform.system() == "Windows":
        return True

def is_linux():
    if platform.system() == "Linux":
        return True

def is_odd(n):
    return n % 2 != 0

def is_even(n):
    return n % 2 == 0

def get_image_size(image):
    if is_numpy(image):
        h, w = image.shape[:2]
        return h, w

    if is_pil(image):
        w, h = image.size
        return h, w

    if is_tensor(image):
        if len(image.shape) == 4 or len(image.shape) == 3:
            w, h = image.shape[-2:]
            return h, w
    else:
        raise ValueError("Unsupported input type")

def get_image_num_channels(img):
    if is_tensor(img):
        if img.ndim == 2:
            return 1
        elif img.ndim > 2:
            return img.shape[-3]

    if is_pil(img):
        return 1 if img.mode == 'L' else 3

    if is_numpy(img):
        return 1 if is_gray_image(img) else 3

if __name__=="__main__":
    image_name = r'D:\PythonProject\pyzjrPyPi\pyzjr\augmentation\test.png'
    print(is_file(image_name), is_windows())