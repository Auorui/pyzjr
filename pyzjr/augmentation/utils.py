import numpy as np
import torch

def get_shape(image):
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            rows, cols = image.shape
            return rows, cols
        else:
            raise ValueError("[pyzjr]:The input NumPy array must be two-dimensional")

    if torch.is_tensor(image):
        if len(image.shape) == 4:
            rows, cols = image.shape[-2:]
            return rows, cols
        else:
            raise ValueError("[pyzjr]:The input PyTorch tensor must be four-dimensional")

    raise ValueError("[pyzjr]:Unsupported input type")

def is_rgb_image(image):
    return len(image.shape) == 3 and image.shape[-1] == 3

def is_gray_image(image):
    return (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1)

def get_num_channels(image):
    return image.shape[2] if len(image.shape) == 3 else 1

def not_rgb_warning(image):
    if not is_rgb_image(image):
        message = "This transformation expects 3-channel images"
        if is_gray_image(image):
            message += "\nYou can convert your gray image to RGB using cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))"
        raise ValueError(message)

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
