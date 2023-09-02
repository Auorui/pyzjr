import numpy as np
from PIL import Image

def cvtColor(image):
    """转化为RGB格式"""
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        img = image.convert('RGB')
        return img

def show_config(**kwargs):
    """显示配置"""
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def normalize_image(image):
    """
    将图像归一化到 [0, 1] 的范围
    :param image: 输入的图像
    :return: 归一化后的图像
    """
    normalized_image = image / 255.0
    return normalized_image

def resizepad_image(image, size):
    """
    将调整图像大小并进行灰度填充
    :param image: 输入图像, PIL Image 对象
    :param size: 目标尺寸，形如 (width, height)
    :return: 调整大小后的图像，PIL Image 对象
    """
    iw, ih = image.size
    w, h = size

    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh