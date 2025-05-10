"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is used for color space conversion.
"""
import colorsys
import cv2
import numpy as np
from PIL import Image
from pyzjr.utils.check import is_numpy, is_pil

def pil2cv(pil_image):
    """将PIL图像转换为OpenCV图像"""
    if pil_image.mode == 'L':
        open_cv_image = np.array(pil_image)
    else:
        open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return open_cv_image

def cv2pil(cv_image):
    """将OpenCV图像转换为PIL图像"""
    if cv_image.ndim == 2:
        pil_image = Image.fromarray(cv_image, mode='L')
    else:
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    return pil_image

def to_gray(image):
    if is_pil(image):
        return image.convert('L')
    elif is_numpy(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise TypeError("Unsupported image type. Expected PIL Image or OpenCV (NumPy array).")

def bgr2rgb(image):
    if is_pil(image):
        image_np = np.asarray(image)
        image_rgb = image_np[:, :, ::-1]
        return Image.fromarray(image_rgb).convert("RGB")
    elif is_numpy(image):
        return image[:, :, ::-1]
    else:
        raise TypeError("Unsupported image type. Expected PIL Image or OpenCV (NumPy array).")

def rgb2bgr(image):
    if is_pil(image):
        image_np = np.asarray(image)
        image_bgr = image_np[:, :, ::-1]
        return Image.fromarray(image_bgr)
    elif is_numpy(image):
        return image[:, :, ::-1]
    else:
        raise TypeError("Unsupported image type. Expected PIL Image or OpenCV (NumPy array).")

def to_hsv(image):
    if is_pil(image):
        image_np = np.asarray(image)
        image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        return Image.fromarray(image_hsv)
    elif is_numpy(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        raise TypeError("Unsupported image type. Expected PIL Image or OpenCV (NumPy array).")

def hsv2rgb(image):
    if is_pil(image):
        image_np = np.asarray(image)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_HSV2RGB)
        return Image.fromarray(image_rgb)
    elif is_numpy(image):
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    else:
        raise TypeError("Unsupported image type. Expected PIL Image or OpenCV (NumPy array).")

def hsv2bgr(image):
    if is_pil(image):
        image_np = np.asarray(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_HSV2BGR)
        return Image.fromarray(image_bgr)
    elif is_numpy(image):
        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    else:
        raise TypeError("Unsupported image type. Expected PIL Image or OpenCV (NumPy array).")


def create_palette(num_classes):
    """
    创建一个颜色调色板，用于可视化不同类别的颜色。
    - 如果num_classes <= 21，使用 VOC 数据集的预定义颜色。
    - 如果num_classes > 21，使用 HSV 颜色空间生成均匀分布的颜色。
    """
    if num_classes <= 21:
        colors = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                  [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                  [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                  [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                  [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                  [0, 64, 128]]
    else:
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    return colors

if __name__=="__main__":
    image_path = r"E:\PythonProject\pyzjrPyPi\pyzjr\test.png"
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
    cv2.imshow('Gray Image', image_cv)
    # gray_cv = to_gray(image_cv)
    # cv2.imshow('Gray Image', gray_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
