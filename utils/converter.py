import cv2
import numpy as np
from PIL import Image
from itertools import repeat

from pyzjr.utils.check import is_Iterable

def _ntuple(n):
    def parse(x):
        if is_Iterable(x):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

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



