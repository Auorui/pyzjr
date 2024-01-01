import numpy as np
import random
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
import cv2

from pyzjr.Graphmath.math import rand

__all__ = ["pad_if_smaller", "Compose", "RandomResize", "RandomHorizontalFlip", "RandomVerticalFlip",
           "RandomCrop", "CenterCrop", "ToTensor", "Normalize", "ToHsv", "RandomContrast", "RandomBrightness",]

def pad_if_smaller(img, size, fill=0):
    """如果图像最小边长小于给定size，则用数值fill进行padding"""
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, [0, 0, padw, padh], fill=fill)
    return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, [size])
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, [size], interpolation=Image.NEAREST)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class ToHsv(object):
    def __init__(self, hue_range=(-0.1, 0.1), saturation_range=(0.6, 1.4), value_range=(0.7, 1.3)):
        self.hue_range = hue_range
        self.saturation_range = saturation_range
        self.value_range = value_range

    def __call__(self, image, target):
        hsv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)

        hue_factor = rand(self.hue_range[0], self.hue_range[1])
        saturation_factor = rand(self.saturation_range[0], self.saturation_range[1])
        value_factor = rand(self.value_range[0], self.value_range[1])

        hsv_image[:, :, 0] = np.clip(hsv_image[:, :, 0] * hue_factor, 0, 179)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * value_factor, 0, 255)

        rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

        return rgb_image, target

class RandomContrast(object):
    def __init__(self, contrast_range=(0.5, 2.0)):
        self.contrast_range = contrast_range

    def __call__(self, image, target):
        contrast_factor = rand(self.contrast_range[0], self.contrast_range[1])
        image = F.adjust_contrast(image, contrast_factor)

        return image, target

class RandomBrightness(object):
    def __init__(self, brightness_range=(0.5, 2.0)):
        self.brightness_range = brightness_range

    def __call__(self, image, target):
        brightness_factor = rand(self.brightness_range[0], self.brightness_range[1])
        image = F.adjust_brightness(image, brightness_factor)

        return image, target


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image = Image.open(r'./VOCdevkit\VOC2007\JPEGImages\2007_000032.jpg')
    target = Image.open(r'./VOCdevkit\VOC2007\SegmentationClass\2007_000032.png')

    # 创建数据增强操作的组合
    transforms = Compose([
        RandomResize(min_size=256, max_size=512),
        RandomHorizontalFlip(flip_prob=0.5),
        RandomCrop(size=224),
        CenterCrop(size=200),
        ToHsv(hue_range=(-0.1, 0.1), saturation_range=(0.6, 1.4), value_range=(0.7, 1.3)),

        ToTensor(),

        RandomContrast(),
        RandomBrightness(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transformed_image, transformed_target = transforms(image, target)
    transformed_image = np.clip(transformed_image, 0, 1)
    plt.imshow(transformed_image.permute(1, 2, 0))
    plt.axis('off')
    plt.show()