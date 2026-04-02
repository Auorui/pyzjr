"""
Copyright (c) 2022, Auorui.
All rights reserved.

Call torchvision for image enhancement
"""
import random
import torch
import torchvision
import numpy as np
import PIL.Image as Image
from pyzjr.utils.randfun import rand
import torchvision.transforms.functional as tf
from pyzjr.augmentation.transforms.apply import Images

def pad_if_smaller(img, size, fill=0):
    """If the minimum edge length of the image is smaller than the given size, use a numerical fill for padding"""
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = tf.pad(img, [0, 0, padw, padh], fill=fill)
    return img

class tvisionToTensor(object):
    def __call__(self, image, target):
        image = tf.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64).permute(2, 0, 1)
        return image, target

class tvisionRandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # The size input here is of type int, so the minimum edge length of the image is scaled to size
        image = tf.resize(image, [size])
        # Note that the interpolation here is only available after torchvision (0.9.0) in InterpolationMode. NEAREST
        # If it is a previous version, PIL.ImageNEAREST needs to be used
        if torchvision.__version__>= "0.9.0":
            NEAREST = torchvision.transforms.InterpolationMode.NEAREST
        else:
            NEAREST = Images.NEAREST
        target = tf.resize(target, [size], interpolation=NEAREST)
        return image, target

class tvisionRandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = tf.hflip(image)
            target = tf.hflip(target)
        return image, target

class tvisionRandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = tf.vflip(image)
            target = tf.vflip(target)
        return image, target

class tvisionRandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=0)
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

class tvisionPerspectiveTransform:
    def __init__(self, distortion_scale=0.5, p=0.5, fill=None):
        self.distortion_scale = distortion_scale
        self.p = p
        self.fill = fill

    def __call__(self, img, target):
        if random.random() < self.p:
            width, height = img.size
            half_h = height // 2
            half_w = width // 2
            topleft = [
                random.randint(0, int(self.distortion_scale * half_w)),
                random.randint(0, int(self.distortion_scale * half_h))
            ]
            topright = [
                random.randint(width - int(self.distortion_scale * half_w), width - 1),
                random.randint(0, int(self.distortion_scale * half_h))
            ]
            botright = [
                random.randint(width - int(self.distortion_scale * half_w), width - 1),
                random.randint(height - int(self.distortion_scale * half_h), height - 1)
            ]
            botleft = [
                random.randint(0, int(self.distortion_scale * half_w)),
                random.randint(height - int(self.distortion_scale * half_h), height - 1)
            ]
            startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
            endpoints = [topleft, topright, botright, botleft]
            img = tf.perspective(img, startpoints, endpoints,
                                 interpolation=tf.InterpolationMode.BILINEAR, fill=self.fill)
            target = tf.perspective(target, startpoints, endpoints,
                                    interpolation=tf.InterpolationMode.NEAREST, fill=self.fill)
        return img, target

class tvisionColorJitter:
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, target):
        if self.brightness > 0:
            brightness_factor = rand(1-self.brightness, 1+self.brightness)
            image = tf.adjust_brightness(image, brightness_factor)
        if self.contrast > 0:
            contrast_factor = rand(1-self.contrast, 1+self.contrast)
            image = tf.adjust_contrast(image, contrast_factor)
        if self.saturation > 0:
            saturation_factor = rand(1-self.saturation, 1+self.saturation)
            image = tf.adjust_saturation(image, saturation_factor)
        if self.hue > 0:
            hue_factor = rand(-self.hue, self.hue)
            image = tf.adjust_hue(image, hue_factor)

        return image, target

class tvisionResizePad:
    def __init__(self, size, fill=128, label_fill=0):
        self.size = (size, size) if isinstance(size, int) else size
        self.fill = (fill, fill, fill) if isinstance(fill, int) else fill
        self.label_fill = (label_fill, label_fill, label_fill) if isinstance(label_fill, int) else label_fill

    def __call__(self, image, target):
        iw, ih = image.size
        w, h = self.size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        image = image.resize((nw,nh), Images.CUBIC)
        new_image = Image.new('RGB', [w, h], self.fill)
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        target = target.resize((nw,nh), Images.NEAREST)
        new_target = Image.new('RGB', [w, h], self.label_fill)
        new_target.paste(target, ((w-nw)//2, (h-nh)//2))
        return new_image, new_target


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pyzjr.augmentation.transforms.apply import Compose
    from pyzjr.visualize.io.matplot import matplotlib_patch
    matplotlib_patch()
    image_path = r'E:\PythonProject\Pytorch_Segmentation_Auorui\data\VOCdevkit\VOC2012\JPEGImages\2007_000033.jpg'
    mask_path = r'E:\PythonProject\Pytorch_Segmentation_Auorui\data\VOCdevkit\VOC2012\SegmentationClass\2007_000033.png'

    # 创建数据增强操作的组合
    transforms = Compose([
        # tvisionRandomResize(min_size=256, max_size=512),
        # tvisionRandomHorizontalFlip(prob=0.5),
        # tvisionRandomCrop(size=256),
        # tvisionCenterCrop(size=200),
        # tvisionColorJitter(),
        tvisionResizePad(size=256),
        # tvisionPerspectiveTransform(0.6),
        tvisionToTensor(),
        # tvisionRandomContrast(),
        # tvisionRandomBrightness(),
        # tvisionNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')
    trans_img, trans_mask = transforms(img, mask)
    alpha = 0.5
    fused = alpha * trans_img.permute(1, 2, 0) + (1 - alpha) * trans_mask.permute(1, 2, 0)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Transformed Image")
    plt.imshow(trans_img.permute(1, 2, 0))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Transformed Mask")
    plt.imshow(trans_mask.permute(1, 2, 0))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Fused Image & Mask")
    plt.imshow(fused)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
