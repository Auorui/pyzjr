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

def random_apply(img, transforms: list, prob, label=None):
    """
    以给定的概率随机应用transforms列表，支持img和label同时传入
    Args:
        img: Image to be randomly applied a list transformations.
        transforms (list): List of transformations to be applied.
        prob (float): The probability to apply the transformation list.
        label: (optional) Corresponding label/mask for segmentation etc.

    Returns:
        Transformed image (and label if provided).
    """
    if prob < random.random():
        return (img, label) if label is not None else img
    if label is not None:
        for transform in transforms:
            img, label = transform(img, label)
        return img, label
    else:
        for transform in transforms:
            img = transform(img)
        return img

def random_order(img, transforms: list, label=None):
    """
    以随机顺序应用transforms列表，支持img和label同时传入
    Args:
        img: Image to be applied transformations in a random order.
        transforms (list): List of the transformations to be applied.
        label: (optional) Corresponding label/mask.

    Returns:
        Transformed image (and label if provided).
    """
    random.shuffle(transforms)
    if label is not None:
        for transform in transforms:
            img, label = transform(img, label)
        return img, label
    else:
        for transform in transforms:
            img = transform(img)
        return img

def random_choice(img, transforms: list, label=None):
    """
    从transforms列表中随机选择一个变换，并将其应用于图像，支持img和label同时传入
    Args:
        img: Image to be applied transformation.
        transforms (list): List of transformations to be chosen from to apply.
        label: (optional) Corresponding label/mask.

    Returns:
        Transformed image (and label if provided).
    """
    t = random.choice(transforms)
    if label is not None:
        return t(img, label)
    else:
        return t(img)

def uniform_augment(img, transforms, num_ops, label=None):
    """
    为每个变换随机分配一个概率, 每个图像决定是否应用它，支持img和label同时传入
    Args:
        img: Image to be applied transformation.
        transforms (list): List of transformations to be chosen from to apply.
        num_ops (int): number of transforms to sequentially apply.
        label: (optional) Corresponding label/mask.

    Returns:
        Transformed image (and label if provided).
    """
    op_idx = np.random.choice(len(transforms), size=num_ops, replace=False)
    if label is not None:
        for idx in op_idx:
            augment_op = transforms[idx]
            pr = random.random()
            if random.random() < pr:
                img, label = augment_op(img.copy(), label.copy())
        return img, label
    else:
        for idx in op_idx:
            augment_op = transforms[idx]
            pr = random.random()
            if random.random() < pr:
                img = augment_op(img.copy())
        return img

class Compose(object):
    """Used to combine transforms."""
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

    def __call__(self, image, label):
        return random_apply(image, self.transforms, self.prob, label=label)

class RandomChoice():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        return random_choice(image, self.transforms, label=label)

class RandomOrder():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        return random_order(image, self.transforms, label=label)

class UniformAugment():
    def __init__(self, transforms, num_ops):
        self.transforms = transforms
        self.num_ops = num_ops

    def __call__(self, image, label):
        return uniform_augment(image, self.transforms, self.num_ops, label=label)

class PILtoNdArray:
    def __call__(self, image, label):
        image_np = np.array(image, dtype=np.float32)
        label_np = np.array(label, dtype=np.int64)
        return image_np, label_np

class NdArraytoPIL:
    def __call__(self, image_np, label_np):
        image_np = np.asarray(image_np)
        label_np = np.asarray(label_np)
        image_pil = Image.fromarray(image_np, self.get_mode(image_np))
        label_pil = Image.fromarray(label_np, self.get_mode(label_np))
        return image_pil, label_pil

    def get_mode(self, arr):
        if arr.ndim == 2:
            return "L"
        elif arr.shape[2] == 3:
            return "RGB"
        elif arr.shape[2] == 4:
            return "RGBA"
        else:
            raise ValueError(f"Unsupported shape {arr.shape}")
