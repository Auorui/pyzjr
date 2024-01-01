"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is used for color space conversion.
"""
import numpy as np
from PIL import Image

__all__ = ["grayscale", "rgb2bgr", "bgr2rgb", "bgr2hsv", "hsv2bgr", "rgb2hsv", "hsv2rgb",]

def rgb_to_hsv(r, g, b):
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        return 0.0, 0.0, v
    s = (maxc-minc) / maxc
    rc = (maxc-r) / (maxc-minc)
    gc = (maxc-g) / (maxc-minc)
    bc = (maxc-b) / (maxc-minc)
    if r == maxc:
        h = bc-gc
    elif g == maxc:
        h = 2.0+rc-bc
    else:
        h = 4.0+gc-rc
    h = (h/6.0) % 1.0
    return h, s, v

def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h*6.0) # XXX assume int() truncates!
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q
    # Cannot get here

def grayscale(np_bgr_img):
    """Convert the input image to a grayscale image"""
    img = Image.fromarray(np_bgr_img, 'RGB')
    gray_img = img.convert('L')
    gray_np_img = np.array(gray_img)
    return gray_np_img

def rgb2bgr(np_rgb_img, is_hwc=True):
    """Convert RGB img to BGR img."""
    if is_hwc:
        np_bgr_img = np_rgb_img[:, :, ::-1]
    else:
        np_bgr_img = np_rgb_img[::-1, :, :]
    return np_bgr_img

def bgr2rgb(np_bgr_img, is_hwc=True):
    """Convert BGR img to RGB img."""
    return rgb2bgr(np_bgr_img, is_hwc)


def rgb2hsv(np_rgb_img, is_hwc=True):
    """Convert RGB img to HSV img."""
    if is_hwc:
        r, g, b = np_rgb_img[:, :, 0], np_rgb_img[:, :, 1], np_rgb_img[:, :, 2]
    else:
        r, g, b = np_rgb_img[0, :, :], np_rgb_img[1, :, :], np_rgb_img[2, :, :]
    to_hsv = np.vectorize(rgb_to_hsv)
    h, s, v = to_hsv(r, g, b)
    if is_hwc:
        axis = 2
    else:
        axis = 0
    np_hsv_img = np.stack((h, s, v), axis=axis)
    return np_hsv_img


def hsv2rgb(np_hsv_img, is_hwc=True):
    """
    Convert HSV img to RGB img.
    """
    if is_hwc:
        h, s, v = np_hsv_img[:, :, 0], np_hsv_img[:, :, 1], np_hsv_img[:, :, 2]
    else:
        h, s, v = np_hsv_img[0, :, :], np_hsv_img[1, :, :], np_hsv_img[2, :, :]

    h = (h % 1.0)
    s = np.clip(s, 0, 1)
    v = np.clip(v, 0, 1)
    h = np.round(h * 255).astype(np.uint8)
    s = np.round(s * 255).astype(np.uint8)
    v = np.round(v * 255).astype(np.uint8)

    to_rgb = np.vectorize(hsv_to_rgb)
    r, g, b = to_rgb(h / 255.0, s / 255.0, v / 255.0)  # Normalize to [0, 1]

    if is_hwc:
        axis = 2
    else:
        axis = 0
    np_rgb_img = np.stack((r, g, b), axis=axis)
    return np_rgb_img

def bgr2hsv(np_bgr_img, is_hwc=True):
    """Convert BGR img to HSV img."""
    np_rgb_img = bgr2rgb(np_bgr_img)
    np_hsv_img = rgb2hsv(np_rgb_img, is_hwc=is_hwc)
    return np_hsv_img

def hsv2bgr(np_hsv_img, is_hwc=True):
    """Convert HSV img to BGR img."""
    np_rgb_img = hsv2rgb(np_hsv_img)
    np_bgr_img = rgb2bgr(np_rgb_img,is_hwc=is_hwc)
    return np_bgr_img