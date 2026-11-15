# import pyzjr.constants as const
# -i https://pypi.tuna.tsinghua.edu.cn/simple

from pyzjr.io.colorspace import get_colors

e = 2.718281828459045
pi = 3.141592653589793

IMG_FORMATS = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm')  # include image suffixes
VID_FORMATS = ('asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv')  # include video suffixes

# keyboard
Esc = ESC =  27
Enter = ENTER = 13
Space = 32

# BGR
blue = get_colors('blue')
green = get_colors('green')
red = get_colors('red')
dark_blue = get_colors('dark_blue')
dark_green = get_colors('dark_green')
dark_red = get_colors('dark_red')
blue_green = get_colors('blue_green')
magenta = get_colors('magenta')
black = get_colors('black')
grey = get_colors('grey')
silvery = get_colors('silvery')
white = get_colors('white')
yellow = get_colors('yellow')
orange = get_colors('orange')
purple = get_colors('purple')
violet = get_colors('violet')
brown = get_colors('brown')
pink = get_colors('pink')
# RGB
rgb_blue = get_colors('blue', to_rgb=True)
rgb_green = get_colors('green', to_rgb=True)
rgb_red = get_colors('red', to_rgb=True)
rgb_dark_blue = get_colors('dark_blue', to_rgb=True)
rgb_dark_green = get_colors('dark_green', to_rgb=True)
rgb_dark_red = get_colors('dark_red', to_rgb=True)
rgb_blue_green = get_colors('blue_green', to_rgb=True)
rgb_magenta = get_colors('magenta', to_rgb=True)
rgb_black = get_colors('black', to_rgb=True)
rgb_grey = get_colors('grey', to_rgb=True)
rgb_silvery = get_colors('silvery', to_rgb=True)
rgb_white = get_colors('white', to_rgb=True)
rgb_yellow = get_colors('yellow', to_rgb=True)
rgb_orange = get_colors('orange', to_rgb=True)
rgb_purple = get_colors('purple', to_rgb=True)
rgb_violet = get_colors('violet', to_rgb=True)
rgb_brown = get_colors('brown', to_rgb=True)
rgb_pink = get_colors('pink', to_rgb=True)


if __name__=="__main__":
    import cv2
    import numpy as np

    height, width = 300, 400
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = magenta
    cv2.imshow('Color', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()