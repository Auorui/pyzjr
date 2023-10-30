import numpy as np
from PIL import Image
import torch
import random
from thop import clever_format, profile
from torchsummary import summary

from pyzjr.utils import gpu
from pyzjr.FM import getPhotopath
from pyzjr.core import is_rgb_image

def mixup(img1, labels, img2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return img, labels

def mix_up_single(batch_size, img, label, alpha=0.2):
    def cir_shift(data):
        index = list(range(1, batch_size)) + [0]
        data = data[index, ...]
        return data

    lam = np.random.beta(alpha, alpha, batch_size)
    lam_img = lam.reshape((batch_size, 1, 1, 1))
    mix_img = lam_img * img + (1 - lam_img) * cir_shift(img)

    lam_label = lam.reshape((batch_size, 1))
    mix_label = lam_label * label + (1 - lam_label) * cir_shift(label)

    return mix_img, mix_label


def mix_up_muti(tmp, batch_size, img, label, alpha=0.2):
    lam = np.random.beta(alpha, alpha, batch_size)
    if tmp.is_first:
        lam = np.ones(batch_size)
        tmp.is_first = False

    lam_img = lam.reshape((batch_size, 1, 1, 1))
    mix_img = lam_img * img + (1 - lam_img) * tmp.image

    lam_label = lam.reshape(batch_size, 1)
    mix_label = lam_label * label + (1 - lam_label) * tmp.label
    tmp.image = mix_img
    tmp.label = mix_label

    return mix_img, mix_label

def replicate(im, labels):
    # Replicate labels
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


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

def seed_worker(worker_id):
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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

def compute_mean_std(_dataset, imagedim=0):
    "计算数据集的mean和std"
    data_r = np.dstack([_dataset[i][imagedim][:, :, 0] for i in range(len(_dataset))])
    data_g = np.dstack([_dataset[i][imagedim][:, :, 1] for i in range(len(_dataset))])
    data_b = np.dstack([_dataset[i][imagedim][:, :, 2] for i in range(len(_dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)
    return mean, std
