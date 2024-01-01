import numpy as np
from PIL import Image
import torch

__all__ = ["is_numpy","is_pil","is_tensor","is_rgb_image","is_gray_image","get_num_channels","get_image_size"]


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

def get_num_channels(image):
    return image.shape[2] if len(image.shape) == 3 else 1

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
        raise ValueError("[pyzjr]:Unsupported input type")




