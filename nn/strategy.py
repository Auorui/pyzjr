import numpy as np
import torch
import random
import warnings

# def mixup(img1, labels, img2, labels2):
#     # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
#     r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
#     img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
#     labels = np.concatenate((labels, labels2), 0)
#     return img, labels

def preprocess_input(image, mean=None, std=None):
    """
    将图像预处理为适合神经网络输入的格式。
    如果图像的最大值大于1，则先归一化到 [0, 1] 范围，
    然后根据提供的均值（mean）和标准差（std）进行标准化处理。

    :param image: 输入的图像，可以是 PIL 图像或 NumPy 数组格式。
    :param mean: 可选，图像标准化时使用的均值，默认为 None。
    :param std: 可选，图像标准化时使用的标准差，默认为 None。
    :return: 归一化和标准化后的图像。
    """
    image = np.asarray(image)
    if np.max(image) > 1:
        image = image / 255.0

    if mean is not None:
        mean = np.array(mean)
        image = image - mean

    if std is not None:
        std = np.array(std)
        image = image / std

    return image.astype(np.float32)

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

def colormap2label(image, colormap, dtype="int64"):
    colormap_label = np.zeros(256 ** 3, dtype=np.int64)
    image = np.array(image, dtype=dtype)
    for i, color_map in enumerate(colormap):
        colormap_label[(color_map[0] * 256 + color_map[1]) * 256 + color_map[2]] = i
    idx = ((image[:, :, 0] * 256 + image[:, :, 1]) * 256
           + image[:, :, 2])
    return colormap_label[idx]