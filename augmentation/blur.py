"""
Copyright (c) 2023, Auorui.
All rights reserved.

部分手写实现,仅供学习参考,最好还是使用cv2
https://blog.csdn.net/m0_62919535/category_11936595.html
"""
import cv2
import numpy as np
from pyzjr.utils.check import is_odd, is_gray_image

def meanblur(img, ksize):
    """均值滤波 """
    blur_img = cv2.blur(img, ksize=ksize)
    return blur_img

def medianblur(img, ksize):
    """中值滤波"""
    if img.dtype == np.float32 and ksize not in {3, 5, 7, 9, 11}:
        raise ValueError(f"Invalid ksize value {ksize}.The available values are 3, 5, and 7")
    medblur_img = cv2.medianBlur(img, ksize=ksize)
    return medblur_img

def gaussianblur(img, ksize, sigma=None):
    """
    高斯模糊, 提供给不熟悉高斯模糊参数的用户, sigma根据ksize进行自动计算,具体可以参考下面
    https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
    """
    if sigma is None:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    gaussianblur_img = cv2.GaussianBlur(img, ksize=(ksize, ksize), sigmaX=sigma)
    return gaussianblur_img

def bilateralblur(img, d=5, sigma_color=75, sigma_space=75):
    """双边滤波"""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def medianfilter(image, ksize):
    assert is_odd(ksize), "Ksize must be an odd number"

    if is_gray_image(image):
        height, width = image.shape
        channels = 1
        image = image[:, :, np.newaxis]
    else:
        height, width, channels = image.shape

    half_window = ksize // 2
    padded_image = np.pad(image, ((half_window, half_window), (half_window, half_window), (0, 0)), mode='reflect')
    filtered_image = np.zeros_like(image)

    for c in range(channels):
        channel_data = padded_image[:, :, c]
        shape = (height, width, ksize, ksize)
        strides = (channel_data.strides[0], channel_data.strides[1], channel_data.strides[0], channel_data.strides[1])
        windows = np.lib.stride_tricks.as_strided(channel_data, shape=shape, strides=strides)
        filtered_image[:, :, c] = np.median(windows, axis=(2, 3))
    if channels == 1:
        filtered_image = filtered_image[:, :, 0]
    return filtered_image

def meanfilter(image, ksize):
    assert is_odd(ksize), "Ksize must be an odd number"

    if is_gray_image(image):
        height, width = image.shape
        channels = 1
        image = image[:, :, np.newaxis]
    else:
        height, width, channels = image.shape

    half_window = ksize // 2
    padded_image = np.pad(image, ((half_window, half_window), (half_window, half_window), (0, 0)), mode='reflect')
    filtered_image = np.zeros_like(image)

    for c in range(channels):
        channel_data = padded_image[:, :, c]
        shape = (height, width, ksize, ksize)
        strides = (channel_data.strides[0], channel_data.strides[1], channel_data.strides[0], channel_data.strides[1])
        windows = np.lib.stride_tricks.as_strided(channel_data, shape=shape, strides=strides)
        filtered_image[:, :, c] = np.mean(windows, axis=(2, 3))

    if channels == 1:
        filtered_image = filtered_image[:, :, 0]

    return filtered_image

def gaussian_kernel(size, sigma):
    center = size // 2
    y, x = np.ogrid[-center:center+1, -center:center+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def gaussianfilter(image, ksize, sigma=None):
    """
    sigma 可参考此处
    https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
    """
    assert is_odd(ksize), "Ksize must be an odd number"
    if sigma is None:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8

    if is_gray_image(image):
        height, width = image.shape
        channels = 1
        image = image[:, :, np.newaxis]
    else:
        height, width, channels = image.shape

    half_window = ksize // 2
    padded_image = np.pad(image, ((half_window, half_window), (half_window, half_window), (0, 0)), mode='reflect')
    filtered_image = np.zeros_like(image)
    kernel = gaussian_kernel(ksize, sigma)

    for c in range(channels):
        channel_data = padded_image[:, :, c]
        shape = (height, width, ksize, ksize)
        strides = (channel_data.strides[0], channel_data.strides[1], channel_data.strides[0], channel_data.strides[1])
        windows = np.lib.stride_tricks.as_strided(channel_data, shape=shape, strides=strides)
        filtered_image[:, :, c] = np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))

    if channels == 1:
        filtered_image = filtered_image[:, :, 0]

    return filtered_image

def bilateralfilter(image, d=5, sigma_color=75, sigma_space=75):
    def gaussian(x, sigma):
        return np.exp(-(x**2) / (2 * sigma**2))

    assert is_odd(d), "Diameter must be an odd number"

    if is_gray_image(image):
        height, width = image.shape
        channels = 1
        image = image[:, :, np.newaxis]
    else:
        height, width, channels = image.shape

    half_d = d // 2
    padded_image = np.pad(image, ((half_d, half_d), (half_d, half_d), (0, 0)), mode='reflect')
    filtered_image = np.zeros_like(image)

    y, x = np.ogrid[-half_d:half_d+1, -half_d:half_d+1]
    spatial_weights = gaussian(np.sqrt(x**2 + y**2), sigma_space)

    for c in range(channels):
        channel_data = padded_image[:, :, c]

        for i in range(height):
            for j in range(width):
                neighborhood = channel_data[i:i+d, j:j+d]
                color_weights = gaussian(neighborhood - channel_data[i + half_d, j + half_d], sigma_color)
                weights = spatial_weights * color_weights
                weights /= np.sum(weights)
                filtered_pixel = np.sum(weights * neighborhood)
                filtered_image[i, j, c] = filtered_pixel

    if channels == 1:
        filtered_image = filtered_image[:, :, 0]

    return filtered_image

if __name__=="__main__":
    import pyzjr
    image = pyzjr.bgr_read("dog.png", is_gray=False)
    with pyzjr.Runcodes():
        image = bilateralfilter(image)
    pyzjr.display(image)