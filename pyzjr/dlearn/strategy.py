import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import random
from pyzjr.augmentation.mask_ops import convert_np
import warnings

# def mixup(img1, labels, img2, labels2):
#     # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
#     r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
#     img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
#     labels = np.concatenate((labels, labels2), 0)
#     return img, labels
#
# def mix_up_single(batch_size, img, label, alpha=0.2):
#     def cir_shift(data):
#         index = list(range(1, batch_size)) + [0]
#         data = data[index, ...]
#         return data
#
#     lam = np.random.beta(alpha, alpha, batch_size)
#     lam_img = lam.reshape((batch_size, 1, 1, 1))
#     mix_img = lam_img * img + (1 - lam_img) * cir_shift(img)
#
#     lam_label = lam.reshape((batch_size, 1))
#     mix_label = lam_label * label + (1 - lam_label) * cir_shift(label)
#
#     return mix_img, mix_label
#
#
# def mix_up_muti(tmp, batch_size, img, label, alpha=0.2):
#     lam = np.random.beta(alpha, alpha, batch_size)
#     if tmp.is_first:
#         lam = np.ones(batch_size)
#         tmp.is_first = False
#
#     lam_img = lam.reshape((batch_size, 1, 1, 1))
#     mix_img = lam_img * img + (1 - lam_img) * tmp.image
#
#     lam_label = lam.reshape(batch_size, 1)
#     mix_label = lam_label * label + (1 - lam_label) * tmp.label
#     tmp.image = mix_img
#     tmp.label = mix_label
#
#     return mix_img, mix_label

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

def image_to_bchw(image_data):
    """
    将图像加载为可输入网络的形状 b c h w, 且 b = 1
    """
    image_np = preprocess_input(convert_np(image_data))
    image_bchw = np.expand_dims(np.transpose(image_np, (2, 0, 1)), 0)
    return image_bchw

def bchw_to_image(image_bchw):
    """
    将网络输出转为图像类型, 且 b = 1
    """
    image_chw = image_bchw[0]  # chw
    image_hwc = image_chw.permute(1, 2, 0)
    image_np = F.softmax(image_hwc, dim=-1).cpu().numpy()
    return image_np

def preprocess_input(image):
    """
    将图像归一化到 [0, 1] 的范围
    :param image: 输入的图像
    :param open: 是否需要打开图像文件（默认为 False）
    :return: 归一化后的图像
    """
    if np.max(np.array(image)) > 1:
        normalized_image = np.asarray(image) / 255.0
        return normalized_image
    else:
        return image


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
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w-nw) // 2, (h-nh) // 2))
        return new_image, nw, nh
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
        return new_image

def SeedEvery(seed=11, rank=0, use_deterministic_algorithms=None):
    """
    :param seed: 设置基础随机种子
    :param rank: 进程的排名，用于为每个进程设置不同的种子
    :param use_deterministic_algorithms: 是否使用确定性算法
    """
    combined_seed = seed + rank
    random.seed(combined_seed)
    np.random.seed(combined_seed)
    torch.manual_seed(combined_seed)
    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        import torch.backends.cuda as cuda
        cuda.matmul.allow_tf32 = True
        cudnn.benchmark = True
        torch.cuda.manual_seed(combined_seed)
        torch.cuda.manual_seed_all(combined_seed)

    if torch.backends.flags_frozen():
        warnings.warn("PyTorch global flag support of backends is disabled, enable it to set global `cudnn` flags.")
        torch.backends.__allow_nonbracketed_mutation_flag = True
        
    if use_deterministic_algorithms:
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
        elif hasattr(torch, "set_deterministic"):
            torch.set_deterministic(True)
        else:
            warnings.warn("If use_deterministic-algorithms=True is set, pytorch version "
                          "greater than 1.8 may be required to use this feature properly.")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
