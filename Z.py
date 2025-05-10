# import pyzjr.Z as Z
# -i https://pypi.tuna.tsinghua.edu.cn/simple
import numpy as np

def getColor(color_name, to_rgb=False):
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

IMG_FORMATS = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm')  # include image suffixes
VID_FORMATS = ('asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv')  # include video suffixes

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
blue = getColor('blue')
green = getColor('green')
red = getColor('red')
dark_blue = getColor('dark_blue')
dark_green = getColor('dark_green')
dark_red = getColor('dark_red')
blue_green = getColor('blue_green')
magenta = getColor('magenta')
black = getColor('black')
grey = getColor('grey')
silvery = getColor('silvery')
white = getColor('white')
yellow = getColor('yellow')
orange = getColor('orange')
purple = getColor('purple')
violet = getColor('violet')
brown = getColor('brown')
pink = getColor('pink')
# RGB
rgb_blue = getColor('blue', to_rgb=True)
rgb_green = getColor('green', to_rgb=True)
rgb_red = getColor('red', to_rgb=True)
rgb_dark_blue = getColor('dark_blue', to_rgb=True)
rgb_dark_green = getColor('dark_green', to_rgb=True)
rgb_dark_red = getColor('dark_red', to_rgb=True)
rgb_blue_green = getColor('blue_green', to_rgb=True)
rgb_magenta = getColor('magenta', to_rgb=True)
rgb_black = getColor('black', to_rgb=True)
rgb_grey = getColor('grey', to_rgb=True)
rgb_silvery = getColor('silvery', to_rgb=True)
rgb_white = getColor('white', to_rgb=True)
rgb_yellow = getColor('yellow', to_rgb=True)
rgb_orange = getColor('orange', to_rgb=True)
rgb_purple = getColor('purple', to_rgb=True)
rgb_violet = getColor('violet', to_rgb=True)
rgb_brown = getColor('brown', to_rgb=True)
rgb_pink = getColor('pink', to_rgb=True)


if __name__=="__main__":
    import cv2
    import numpy as np

    height, width = 300, 400
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (255, 0, 255)
    cv2.imshow('Color', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()