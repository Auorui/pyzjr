"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is based on cv2 implemented transformations
"""
import cv2
import torch
import random
import numpy as np
from pyzjr.augmentation.augments import Centerzoom, horizontal_flip, vertical_flip, adjust_brightness_cv2, \
    adjust_gamma, augment_Hsv, hist_equalize, random_rotation, random_lighting, random_crop, random_resize_crop
from pyzjr.augmentation.blur import (
    meanblur,
    medianblur,
    gaussianblur,
    bilateralblur,
)
from pyzjr.core.error import _check_img_is_opencv
from pyzjr.core.helpers import convert_to_tuple
from pyzjr.Math import rand

__all__ = ["OpencvToTensor",
           "OpencvResize",
           "OpencvSquareResize",
           "OpencvCenterzoom",
           "OpencvHorizontalFlip",
           "OpencvVerticalFlip",
           "OpencvBrightness",
           "OpencvAdjustGamma",
           "OpencvToHSV",
           "OpencvHistEqualize",
           "OpencvRotation",
           "OpencvLighting",
           "OpencvRandomBlur",
           "OpencvCrop",
           "OpencvResizeCrop",
           "OpencvPadResize",
           "OpencvGrayscale",]

class OpencvToTensor():
    """OpencvToTensor类用于BGR格式的ToTensor"""
    def __init__(self, half=False):
        super().__init__()
        self.half = half

    def __call__(self, opencv_img):
        _check_img_is_opencv(opencv_img)
        if len(opencv_img.shape) == 3:  # BGR图像
            img = np.ascontiguousarray(opencv_img.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        else:  # 灰度图像
            img = np.ascontiguousarray(opencv_img)
        img = torch.from_numpy(img)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0
        return img

class OpencvResize():
    """OpencvResize类用于调整OpenCV图像的大小"""
    def __init__(self, size):
        self.size = convert_to_tuple(size)
        self.interp_method = cv2.INTER_CUBIC

    def __call__(self, opencv_img):
        _check_img_is_opencv(opencv_img)
        image_shape = opencv_img.shape
        if image_shape[-1] > self.size[1]:
            self.interp_method = cv2.INTER_AREA
        image = cv2.resize(opencv_img, self.size, interpolation=self.interp_method)
        return image

class OpencvSquareResize():
    """OpencvSquareResize类用于将图像裁剪为正方形并调整大小"""
    def __init__(self, size):
        super().__init__()
        self.h, self.w = convert_to_tuple(size)

    def __call__(self, opencv_img):  # im = np.array HWC
        _check_img_is_opencv(opencv_img)
        imh, imw = opencv_img.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(opencv_img[top:top + m, left:left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)

class OpencvCenterzoom():
    """OpencvCenterzoom类用于中心缩放"""
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def __call__(self, opencv_img):
        _check_img_is_opencv(opencv_img)
        return Centerzoom(opencv_img, self.factor)

class OpencvHorizontalFlip():
    """OpencvCenterzoom类用于随机水平翻转"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.random_float = random.random()

    def __call__(self, opencv_img):
        _check_img_is_opencv(opencv_img)
        if self.random_float >= self.p:
            opencv_img = horizontal_flip(opencv_img)
        return opencv_img

class OpencvVerticalFlip():
    """OpencvCenterzoom类用于随机垂直翻转"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.random_float = random.random()

    def __call__(self, opencv_img):
        _check_img_is_opencv(opencv_img)
        if self.random_float >= self.p:
            opencv_img = vertical_flip(opencv_img)
        return opencv_img

class OpencvBrightness():
    """OpencvBrightness类用于亮度调整"""
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def __call__(self, opencv_img):
        _check_img_is_opencv(opencv_img)
        opencv_img = adjust_brightness_cv2(opencv_img, self.factor)
        return opencv_img

class OpencvAdjustGamma():
    """OpencvAdjustGamma类用于调整图像的gamma值"""
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, opencv_img):
        _check_img_is_opencv(opencv_img)
        opencv_img = adjust_gamma(opencv_img, self.gamma)
        return opencv_img


class OpencvToHSV():
    """OpencvToHSV类用于转换图像到HSV颜色空间"""
    def __init__(self, hue_range=(-0.1, 0.3), saturation_range=(0.5, 1.4), value_range=(0.7, 1.3)):
        self.hgain = rand(hue_range[0], hue_range[1])
        self.sgain = rand(saturation_range[0], saturation_range[1])
        self.vgain = rand(value_range[0], value_range[1])

    def __call__(self, opencv_img):
        _check_img_is_opencv(opencv_img)
        opencv_img = augment_Hsv(opencv_img, self.hgain, self.sgain, self.vgain)
        return opencv_img

class OpencvHistEqualize():
    """OpencvHistEqualize类用于对图像进行直方图均衡化"""
    def __init__(self, clahe=True):
        self.clahe = clahe

    def __call__(self, opencv_img):
        _check_img_is_opencv(opencv_img)
        opencv_img = hist_equalize(opencv_img, self.clahe)
        return opencv_img

class OpencvRotation():
    """OpencvRotation类用于对图像进行随机旋转"""
    def __init__(self, degrees=None):
        self.degrees = degrees

    def __call__(self, opencv_img):
        _check_img_is_opencv(opencv_img)
        opencv_img = random_rotation(opencv_img, self.degrees)
        return opencv_img

class OpencvLighting():
    """OpencvLighting类用于对图像进行随机光照调整"""
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, opencv_img):
        _check_img_is_opencv(opencv_img)
        opencv_img = random_lighting(opencv_img, self.alpha)
        return opencv_img

class OpencvRandomBlur():
    """OpencvRandomBlur类用于对图像进行随机模糊(支持四选一)"""
    def __init__(self, ksize, choice=None):
        self.ksize = ksize
        self.choice = choice
        if self.ksize not in {1, 3, 5, 7, 9, 11}:
            raise ValueError(f"Invalid ksize value {ksize}.The available values are 1, 3, 5, 7, 9, 11")
        if self.choice is None:
            filter_functions = ["blur", "median", "gaussian", "bilateral"]
            self.choice = random.choice(filter_functions)

    def __call__(self, opencv_img):
        _check_img_is_opencv(opencv_img)
        if self.choice == "blur":
            self.ksize = convert_to_tuple(self.ksize)
            opencv_img = meanblur(opencv_img, self.ksize)
        elif self.choice == "median":
            opencv_img = medianblur(opencv_img, self.ksize)
        elif self.choice == "gaussian":
            opencv_img = gaussianblur(opencv_img, ksize=self.ksize)
        elif self.choice == "bilateral":
            opencv_img = bilateralblur(opencv_img, self.ksize)
        return opencv_img

class OpencvCrop():
    """OpencvCrop类用于对图像进行随机裁剪"""
    def __init__(self, size):
        self.size = convert_to_tuple(size)

    def __call__(self, opencv_img):
        _check_img_is_opencv(opencv_img)
        height, width = opencv_img.shape[:2]
        if self.size[0] > height or self.size[1] > width:
            raise ValueError("The cropped size cannot exceed the original image size")
        opencv_img = random_crop(opencv_img, self.size)
        return opencv_img

class OpencvResizeCrop():
    """OpencvResizeCrop类用于对图像进行随机缩放和裁剪"""
    def __init__(self, size, scale_range=(1.5, 2.5)):
        self.size = convert_to_tuple(size)
        self.scale_range = scale_range

    def __call__(self, opencv_img):
        _check_img_is_opencv(opencv_img)
        opencv_img = random_resize_crop(opencv_img, self.size, self.scale_range)
        return opencv_img

class OpencvPadResize():
    """OpencvPadResize类用于调整OpenCV图像的大小并在背景中居中"""
    def __init__(self, size):
        self.size = convert_to_tuple(size)
        self.full = (128, 128, 128)

    def __call__(self, opencv_img):
        _check_img_is_opencv(opencv_img)
        height, width = opencv_img.shape[:2]
        h, w = self.size
        scale = min(w / width, h / height)
        nw = int(width * scale)
        nh = int(height * scale)
        opencv_img = cv2.resize(opencv_img, (nw, nh))
        background = np.full((h, w, 3), self.full, dtype=np.uint8)
        x_offset = (w - nw) // 2
        y_offset = (h - nh) // 2
        background[y_offset:y_offset+nh, x_offset:x_offset+nw] = opencv_img

        return background

class OpencvGrayscale():
    """OpencvGrayscale类用于将BGR图像转为灰度图再转回BGR,一般会导致颜色信息丢失"""
    def __call__(self, opencv_img):
        _check_img_is_opencv(opencv_img)
        opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
        opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_GRAY2BGR)
        return opencv_img

if __name__=="__main__":
    from pyzjr.augmentation.transforms._utils import Compose
    img = cv2.imread(r"D:\PythonProject\pyzjrPyPi\pyzjr\augmentation\test.png")
    transforms = Compose([
        OpencvCenterzoom(1.1),
        OpencvResize(256),
        # OpencvGrayscale(),
        OpencvBrightness(1.4),
        OpencvHorizontalFlip(p=0.5),
        OpencvVerticalFlip(p=0.5),
        OpencvHistEqualize(),
        OpencvAdjustGamma(1.2),
        OpencvLighting(1.2),
        OpencvRotation(),
        OpencvRandomBlur(3, choice="gaussian"),
        OpencvPadResize(512),
        OpencvCrop(300),
        OpencvResizeCrop(200),
        OpencvToTensor(),
    ])
    transformed_image = transforms(img)
    print(transformed_image.shape)
    # 将RGB转为BGR
    transformed_image_bgr = torch.flip(transformed_image, dims=[0])
    img = (transformed_image.permute((1, 2, 0)).numpy() * 255).astype(np.uint8)
    cv2.imshow("ss", img)
    cv2.waitKey(0)




