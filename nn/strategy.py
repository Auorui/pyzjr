import cv2
import numpy as np
from PIL import Image
import torch
import random
import warnings
from pyzjr.utils.check import is_pil, is_numpy

# def mixup(img1, labels, img2, labels2):
#     # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
#     r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
#     img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
#     labels = np.concatenate((labels, labels2), 0)
#     return img, labels

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
