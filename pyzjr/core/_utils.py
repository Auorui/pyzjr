import torch
import numpy as np
from pyzjr.core.general import is_numpy, is_tensor, is_pil, is_gray_image, is_list_or_tuple
from typing import Iterable

def to_numpy(x, dtype=None):
    if is_pil(x):
        return np.array(x, dtype=dtype)
    elif is_tensor(x):
        numpy_array = x.cpu().numpy()
        if dtype is not None:
            numpy_array = numpy_array.astype(dtype)
        return numpy_array
    elif is_numpy(x):
        if dtype is not None:
            return x.astype(dtype)
        return x
    elif isinstance(x, (Iterable, int, float)):
        return np.array(x, dtype=dtype)
    elif is_list_or_tuple(x):
        return np.array(x, dtype=dtype)
    else:
        raise ValueError("Unsupported type")

def to_tensor(x, dtype=None):
    if is_tensor(x):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if is_numpy(x):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if is_list_or_tuple(x):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    else:
        raise ValueError("Unsupported type")

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
        return 1 if is_gray_image else 3