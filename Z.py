# import pyzjr.Z as Z
# -i https://pypi.tuna.tsinghua.edu.cn/simple
import numpy as np

e = 2.718281828459045
pi = 3.141592653589793
half_pi = pi / 2
double_pi = 2 * pi

NUMPY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}

IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm']  # include image suffixes
VID_FORMATS = ['asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv']  # include video suffixes

# 颜色空间转换
BGR2RGB = 4
BGR2HSV = 40
BGR2GRAY = 6
RGB2GRAY = 7
GRAY2BGR = 8
GRAY2RGB = 8
HSV2BGR = 54
HSV2RGB = 55
RGB2HSV = 41
RGB2BGR = 4

# video
Esc = 27
# BGR
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
black = (0, 0, 0)
grey = (192, 192, 192)
white = (255, 255, 255)
yellow = (0, 255, 255)
orange = (0, 97, 255)
purple = (255, 0, 255)
violet = (240, 32, 160)
# RGB
rgb_blue = (0, 0, 255)
rgb_green = (0, 255, 0)
rgb_red = (255, 0, 0)
rgb_black = (0, 0, 0)
rgb_grey = (192, 192, 192)
rgb_white = (255, 255, 255)
rgb_yellow = (255, 255, 0)
rgb_orange = (255, 97, 0)
rgb_purple = (255, 0, 255)
rgb_violet = (160, 32, 240)


VOC_COLOR = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
             [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
             [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
             [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
             [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
             [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']