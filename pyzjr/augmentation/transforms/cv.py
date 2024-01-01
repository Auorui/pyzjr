"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is based on cv2 implemented transformations
"""
import numpy as np
import torch
import cv2

__all__ = ["ToTensor", "ToNumpy", "CenterCrop"]

class OpencvImgOperation():
    def __init__(self):
        super(OpencvImgOperation).__init__()

class ToTensor(OpencvImgOperation):
    """用于BGR的ToTensor"""
    def __init__(self, half=False):
        super().__init__()
        self.half = half

    def __call__(self, img):  # im = np.array HWC in BGR order
        img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        img = torch.from_numpy(img)  # to torch
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0-255 to 0.0-1.0
        return img

class ToNumpy(OpencvImgOperation):
    def __call__(self, img):
        np_img = np.array(img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img

class CenterCrop(OpencvImgOperation):
    # T.Compose([CenterCrop(size), ToTensor()])
    def __init__(self, size=640):
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top:top + m, left:left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)



