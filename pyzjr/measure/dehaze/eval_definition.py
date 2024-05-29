import cv2
import math
import numpy as np
import scipy.ndimage as ndi
from skimage.filters import sobel

def Brenner(img):
    '''
    Brenner梯度函数
    :param img:narray 二维灰度图像
    :return: int 图像越清晰越大
    '''
    shapes = np.shape(img)
    output = 0
    for x in range(0, shapes[0] - 2):
        for y in range(0, shapes[1]):
            output += (int(img[x + 2, y]) - int(img[x, y])) ** 2
    return output


def EOG(img):
    '''
    能量梯度函数(Energy of Gradient)
    :param img:narray 二维灰度图像
    :return: int 图像越清晰越大
    '''
    shapes = np.shape(img)
    output = 0
    for x in range(0, shapes[0] - 1):
        for y in range(0, shapes[1] - 1):
            output += ((int(img[x + 1, y]) - int(img[x, y])) ** 2 + (int(img[x, y + 1]) - int(img[x, y])) ** 2)
    return output

def Roberts(img):
    '''
    Roberts函数
    :param img:narray 二维灰度图像
    :return: int 图像越清晰越大
    '''
    shapes = np.shape(img)
    output = 0
    for x in range(0, shapes[0] - 1):
        for y in range(0, shapes[1] - 1):
            output += ((int(img[x + 1, y + 1]) - int(img[x, y])) ** 2 + (
                    int(img[x + 1, y]) - int(img[x, y + 1])) ** 2)
    return output

def Laplacian(img):
    '''
    Laplace
    :param img:narray 二维灰度图像
    :return: int 图像越清晰越大
    '''
    return cv2.Laplacian(img, cv2.Lap_64F).var()

def SMD(img):
    '''
    SMD(灰度方差)函数
    :param img:narray 二维灰度图像
    :return: int 图像越清晰越大
    '''
    shape = np.shape(img)
    output = 0
    for x in range(1, shape[0] - 1):
        for y in range(0, shape[1]):
            output += math.fabs(int(img[x, y]) - int(img[x, y - 1]))
            output += math.fabs(int(img[x, y] - int(img[x + 1, y])))
    return output

def SMD2(img):
    '''
    （灰度方差乘积）函数
    :param img:narray 二维灰度图像
    :return: int 图像约清晰越大
    '''
    shape = np.shape(img)
    output = 0
    for x in range(0, shape[0] - 1):
        for y in range(0, shape[1] - 1):
            output += math.fabs(int(img[x, y]) - int(img[x + 1, y])) * math.fabs(
                int(img[x, y] - int(img[x, y + 1])))
    return output

def Gradientmetric(img, size=11):
    """
    计算指示图像中模糊强度的度量(0表示无模糊, 1表示最大模糊)
    但实际上还是要根据我们的标准图与模糊图得到模糊区间
    :param img: 单通道进行处理,灰度图
    :param size: 重新模糊过滤器的大小
    :return: float, 唯一不是值越大, 越清晰的算法
    """
    image = np.array(img, dtype=np.float32) / 255
    n_axes = image.ndim
    shape = image.shape
    blur_metric = []

    slices = tuple([slice(2, s - 1) for s in shape])
    for ax in range(n_axes):
        filt_im = ndi.uniform_filter1d(image, size, axis=ax)
        im_sharp = np.abs(sobel(image, axis=ax))
        im_blur = np.abs(sobel(filt_im, axis=ax))
        T = np.maximum(0, im_sharp - im_blur)
        M1 = np.sum(im_sharp[slices])
        M2 = np.sum(T[slices])
        blur_metric.append(np.abs((M1 - M2)) / M1)

    return np.max(blur_metric) if len(blur_metric) > 0 else 0.0
