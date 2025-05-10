"""
Copyright (c) 2022, Auorui.
All rights reserved.

Call torchvision for image enhancement
"""
import random
import torch
import torchvision
import numpy as np
from pyzjr.utils.randfun import rand
import torchvision.transforms.functional as tf
from pyzjr.augmentation.transforms.applytransformer import Images

class tvisionToTensor(object):
    def __call__(self, image, target):
        image = tf.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

def pad_if_smaller(img, size, fill=0):
    """如果图像最小边长小于给定size，则用数值fill进行padding"""
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = tf.pad(img, [0, 0, padw, padh], fill=fill)
    return img

class tvisionRandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = tf.resize(image, [size])
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        if torchvision.__version__>= "0.9.0":
            NEAREST = torchvision.transforms.InterpolationMode.NEAREST
        else:
            NEAREST = Images.NEAREST
        target = tf.resize(target, [size], interpolation=NEAREST)
        return image, target

class tvisionRandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = tf.hflip(image)
            target = tf.hflip(target)
        return image, target

class tvisionRandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = tf.vflip(image)
            target = tf.vflip(target)
        return image, target

class tvisionRandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = torchvision.transforms.RandomCrop.get_params(image, (self.size, self.size))
        image = tf.crop(image, *crop_params)
        target = tf.crop(target, *crop_params)
        return image, target

class tvisionCenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = tf.center_crop(image, self.size)
        target = tf.center_crop(target, self.size)
        return image, target

class tvisionNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = tf.normalize(image, mean=self.mean, std=self.std)
        return image, target

class tvisionRandomBrightness(object):
    def __init__(self, brightness_range=(0.5, 2.0)):
        self.brightness_range = brightness_range

    def __call__(self, image, target):
        brightness_factor = rand(self.brightness_range[0], self.brightness_range[1])
        image = tf.adjust_brightness(image, brightness_factor)

        return image, target

class tvisionRandomContrast(object):
    def __init__(self, contrast_range=(0.5, 2.0)):
        self.contrast_range = contrast_range

    def __call__(self, image, target):
        contrast_factor = rand(self.contrast_range[0], self.contrast_range[1])
        image = tf.adjust_contrast(image, contrast_factor)

        return image, target


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import PIL.Image as Image
    from pyzjr.augmentation.transforms.applytransformer import Compose
    image = Image.open(r'./VOCdevkit\VOC2007\JPEGImages\2007_000032.jpg')
    target = Image.open(r'./VOCdevkit\VOC2007\SegmentationClass\2007_000032.png')

    # 创建数据增强操作的组合
    transforms = Compose([
        tvisionRandomResize(min_size=256, max_size=512),
        tvisionRandomHorizontalFlip(flip_prob=0.5),
        tvisionRandomCrop(size=224),
        tvisionCenterCrop(size=200),
        tvisionToTensor(),

        tvisionRandomContrast(),
        tvisionRandomBrightness(),
        tvisionNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transformed_image, transformed_target = transforms(image, target)
    transformed_image = np.clip(transformed_image, 0, 1)
    plt.imshow(transformed_image.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
