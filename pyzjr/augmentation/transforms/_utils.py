from pyzjr.augmentation.augment import random_apply, random_order, random_choice
from PIL import Image

__all__ = ["Images", "Compose", "RandomApply", "RandomChoice", "RandomOrder"]

class Images:
    if Image.__version__ >= "9.1.0":
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

class Compose():
    """
    Usage Directions:
        Used to combine transforms.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def compose(self,transforms, img):
        for transform in transforms:
            img = transform(img)
        return img

    def __call__(self, *args):
        return self.compose(self.transforms, *args)

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