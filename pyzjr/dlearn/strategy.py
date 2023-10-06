import numpy as np
from PIL import Image
import torch
import random
from thop import clever_format, profile
from torchsummary import summary

from pyzjr.utils import gpu
from pyzjr.FM import getPhotopath
from pyzjr.augmentation.utils import is_rgb_image

def cvtColor(image):
    """Convert to RGB format"""
    if is_rgb_image:
        return image
    else:
        img = image.convert('RGB')
        return img

def normalize_image(image, open=False):
    """
    将图像归一化到 [0, 1] 的范围
    :param image: 输入的图像
    :param open: 是否需要打开图像文件（默认为 False）
    :return: 归一化后的图像
    """
    if open:
        if isinstance(image, str):
            img_opened = Image.open(image)
            image = np.asarray(img_opened)
        else:
            raise ValueError("[pyzjr]:When `open` is True, `image` should be a file path string.")

    normalized_image = image / 255.0
    return normalized_image


def resizepad_image(image, size, frame=True):
    """
    将调整图像大小并进行灰度填充
    :param image: 输入图像, PIL Image 对象
    :param size: 目标尺寸，形如 (width, height)
    :param frame: 是否进行不失真的resize
    :return: 调整大小后的图像，PIL Image 对象
    """
    iw, ih = image.size
    w, h = size
    if frame:
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image, nw, nh
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
        return new_image

def seed_torch(seed=11):
    """
    :param seed:设置随机种子以确保实验的可重现性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def summarys(input_shape, model):
    """
    打印模型的摘要信息，并计算模型的总浮点运算量和总参数数量
    :param input_shape:
    :param model:要进行计算的模型
    """
    device = gpu()
    models = model.to(device)
    summary(models, (3, input_shape[0], input_shape[1]))

    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(models.to(device), (dummy_input, ), verbose=False)
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))

def get_mean_std(image_path):
    """
    Calculate the average and standard deviation of all images under a given path
    Args:
        image_path : pathway of all images
    Return :
        mean : mean value of all the images
        stdev : standard deviation of all pixels
    """
    all_images,_ = getPhotopath(image_path,debug=False)
    num_images = len(all_images)
    mean_sum = std_sum = 0

    for image in all_images:
        img_asarray = normalize_image(image, open=True)
        individual_mean = np.mean(img_asarray)
        individual_stdev = np.std(img_asarray)
        mean_sum += individual_mean
        std_sum += individual_stdev

    mean = mean_sum / num_images
    std = std_sum / num_images
    return mean, std



