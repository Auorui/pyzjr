"""
Copyright (c) 2023, Auorui.
All rights reserved.

Image enhancement application
"""
import numpy as np
import random
from PIL import Image
import PIL

class Images:
    if PIL.__version__ >= "10.0.0":
        FLIP_LEFT_RIGHT = Image.Transpose.FLIP_LEFT_RIGHT
        FLIP_TOP_BOTTOM = Image.Transpose.FLIP_TOP_BOTTOM
        PERSPECTIVE = Image.Transform.PERSPECTIVE
        AFFINE = Image.Transform.AFFINE
        NEAREST = Image.Resampling.NEAREST
        ANTIALIAS = Image.Resampling.LANCZOS
        LINEAR = Image.Resampling.BILINEAR
        CUBIC = Image.Resampling.BICUBIC
    else:
        FLIP_LEFT_RIGHT = Image.FLIP_LEFT_RIGHT
        FLIP_TOP_BOTTOM = Image.FLIP_TOP_BOTTOM
        PERSPECTIVE = Image.PERSPECTIVE
        AFFINE = Image.AFFINE
        NEAREST = Image.NEAREST
        ANTIALIAS = Image.ANTIALIAS
        LINEAR = Image.LINEAR
        CUBIC = Image.CUBIC


def random_apply(img, transforms:list, prob):
    """
    以给定的概率随机应用transforms列表
    Args:
        img: Image to be randomly applied a list transformations.
        transforms (list): List of transformations to be applied.
        prob (float): The probability to apply the transformation list.

    Returns:
        Transformed image.
    """
    if prob < random.random():
        return img
    for transform in transforms:
        img = transform(img)
    return img

def random_order(img, transforms:list):
    """
    以随机顺序应用transforms列表.
    Args:
        img: Image to be applied transformations in a random order.
        transforms (list): List of the transformations to be applied.

    Returns:
        Transformed image.
    """
    random.shuffle(transforms)
    for transform in transforms:
        img = transform(img)
    return img

def random_choice(img, transforms:list):
    """
    从transforms列表中随机选择一个变换，并将其应用于图像。
    Args:
        img: Image to be applied transformation.
        transforms (list): List of transformations to be chosen from to apply.

    Returns:
        Transformed image.
    """
    return random.choice(transforms)(img)

def uniform_augment(img, transforms, num_ops):
    """
    为每个变换随机分配一个概率, 每个图像决定是否应用它。

    Args:
        img: Image to be applied transformation.
        transforms (list): List of transformations to be chosen from to apply.
        num_ops (int): number of transforms to sequentially aaply.

    Returns:
        Transformed image.
    """
    op_idx = np.random.choice(len(transforms), size=num_ops, replace=False)
    for idx in op_idx:
        augment_op = transforms[idx]
        pr = random.random()
        if random.random() < pr:
            img = augment_op(img.copy())
    return img

class Compose(object):
    """
    Usage Directions:
        Used to combine transforms.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label=None):
        if label is not None:
            for t in self.transforms:
                image, label = t(image, label)
            return image, label
        else:
            for t in self.transforms:
                image = t(image)
            return image

class RandomApply():
    def __init__(self, transforms, prob=0.5):
        self.prob = prob
        self.transforms = transforms

    def __call__(self, img):
        return random_apply(img, self.transforms, self.prob)

class RandomChoice():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        return random_choice(img, self.transforms)

class RandomOrder():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        return random_order(img, self.transforms)

class UniformAugment():
    def __init__(self, transforms, num_ops):
        self.transforms = transforms
        self.num_ops = num_ops

    def __call__(self, img):
        return uniform_augment(img, self.transforms, self.num_ops)