"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is used for color space conversion.
"""
import cv2
import colorsys
import numpy as np
from PIL import Image


def to_gray(image):
    if isinstance(image, Image.Image):
        return image.convert('L')
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise TypeError("Unsupported image type. Expected PIL Image or OpenCV (NumPy array).")


def bgr2rgb(image):
    if isinstance(image, Image.Image):
        image_np = np.asarray(image)
        image_rgb = image_np[:, :, ::-1]
        return Image.fromarray(image_rgb).convert("RGB")
    elif isinstance(image, np.ndarray):
        return image[:, :, ::-1].copy()
    else:
        raise TypeError("Unsupported image type. Expected PIL Image or OpenCV (NumPy array).")


def rgb2bgr(image):
    if isinstance(image, Image.Image):
        image_np = np.asarray(image)
        image_bgr = image_np[:, :, ::-1]
        return Image.fromarray(image_bgr)
    elif isinstance(image, np.ndarray):
        return image[:, :, ::-1].copy()
    else:
        raise TypeError("Unsupported image type. Expected PIL Image or OpenCV (NumPy array).")


def to_hsv(image):
    if isinstance(image, Image.Image):
        image_np = np.asarray(image)
        image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        return Image.fromarray(image_hsv)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        raise TypeError("Unsupported image type. Expected PIL Image or OpenCV (NumPy array).")


def hsv2rgb(image):
    if isinstance(image, Image.Image):
        image_np = np.asarray(image)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_HSV2RGB)
        return Image.fromarray(image_rgb)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    else:
        raise TypeError("Unsupported image type. Expected PIL Image or OpenCV (NumPy array).")


def hsv2bgr(image):
    if isinstance(image, Image.Image):
        image_np = np.asarray(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_HSV2BGR)
        return Image.fromarray(image_bgr)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    else:
        raise TypeError("Unsupported image type. Expected PIL Image or OpenCV (NumPy array).")


def create_palette(num_classes):
    """
    Create a color palette for visualizing colors of different categories.

    - If num_classes is less than or equal to 21, use the predefined colors from the VOC dataset.
    - If num_classes exceeds 21, generate uniformly distributed colors using the HSV color space.
    """
    if num_classes <= 21:
        colors = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
                  [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                  [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                  [0, 192, 0], [128, 192, 0], [0, 64, 128]]
    else:
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    return colors


def get_colors(color_name, to_rgb=False):
    bgr_colors = {
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'red': (0, 0, 255),
        "dark_blue": (128, 0, 0),
        'dark_green': (0, 128, 0),
        "dark_red": (0, 0, 128),
        "blue_green": (128, 128, 0),
        'magenta': (255, 0, 255),
        "black": (0, 0, 0),
        "grey": (128, 128, 128),
        "silvery": (192, 192, 192),
        "white": (255, 255, 255),
        "yellow": (0, 255, 255),
        "orange": (0, 97, 255),
        "purple": (255, 0, 255),
        "violet": (240, 32, 160),
        "brown": (19, 69, 139),
        "pink": (203, 192, 255),
    }
    bgr_color = bgr_colors.get(color_name, None)
    if bgr_color is not None and to_rgb:
        return bgr_color[::-1]
    return bgr_color


if __name__=="__main__":
    print(create_palette(28))

    image_path = r"/resources/images/aircraft.jpg"
    image_pil = Image.open(image_path)
    # gray_pil = to_gray(image_pil)
    image_pil = rgb2bgr(image_pil)
    image_pil = bgr2rgb(image_pil)
    image_pil = to_hsv(image_pil)
    image_pil = hsv2rgb(image_pil)
    image_pil.show()

    image_cv = cv2.imread(image_path)
    image_cv = bgr2rgb(image_cv)
    image_cv = rgb2bgr(image_cv)
    image_cv = to_hsv(image_cv)
    image_cv = hsv2bgr(image_cv)

    print(get_colors('red'))

    # cv2.imshow('Gray Image', image_cv)
    # gray_cv = to_gray(image_cv)
    # cv2.imshow('Gray Image', gray_cv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
