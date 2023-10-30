import numpy as np
import torch

__all__=["add_weighted", "normalize_np", "normalization1", "normalization2", "clip","approximate_image","ceilfloor_image",]

def add_weighted(img1, alpha, img2, beta):
    return img1.astype(float) * alpha + img2.astype(float) * beta

def normalize_np(image, mean, denominator=1):
    """零均值化法(中心化),可支持其他的均值化方法,修改denominator"""
    img = image.astype(np.float32)
    img -= mean
    img *= denominator
    return img

def normalization1(image, mean, std):
    image = image / 255  # values will lie between 0 and 1.
    image = (image - mean) / std
    return image

def normalization2(image, max, min):
    image_new = (image - np.min(image))*(max - min)/(np.max(image)-np.min(image)) + min
    return image_new

def approximate_image(image):
    """
    Convert a single channel image into a binary image.
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 only with 255 and 0
    """
    image[image > 127.5] = 255
    image[image < 127.5] = 0
    image = image.astype("uint8")
    return image

def ceilfloor_image(image):
    """
    The pixel value of the input image is limited between the maximum value of 255 and the minimum value of 0
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 with ceilling(maximum 255) and flooring(minimum 0)
    """
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image

def clip(img, dtype, maxval):
    """截断图像的像素值到指定范围，并进行数据类型转换"""
    return np.clip(img, 0, maxval).astype(dtype)

def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr