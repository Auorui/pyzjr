import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from pyzjr.data.utils.path import get_image_path
from pyzjr.utils.check import is_tensor, is_numpy, is_pil, is_gray_image

def get_image_shape(image):
    width, height = 0, 0
    channels = 0
    if is_numpy(image):
        height, width = image.shape[:2]
        channels = 1 if is_gray_image(image) else 3
    elif is_pil(image):
        width, height = image.size
        channels = 1 if image.mode == 'L' else 3
    elif is_tensor(image):
        if len(image.shape) == 4 or len(image.shape) == 3:
            height, width = image.shape[-2:]
            if image.ndim == 2:
                channels = 1
            elif image.ndim > 2:
                channels = image.shape[-3]
    else:
        raise ValueError("Unsupported input type")
    return height, width, channels

def get_dataset_mean_std(train_data):
    train_loader = DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for im, _ in train_loader:
        for d in range(3):
            mean[d] += im[:, d, :, :].mean()
            std[d] += im[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())

def get_images_mean_std(dir_path):
    images_paths = get_image_path(dir_path)
    num_images = len(images_paths)
    mean_sum = np.zeros(3)
    std_sum = np.zeros(3)

    for one_image_path in images_paths:
        pil_image = Image.open(one_image_path).convert("RGB")
        img_asarray = np.asarray(pil_image) / 255.0
        individual_mean = np.mean(img_asarray, axis=(0, 1))
        individual_stdev = np.std(img_asarray, axis=(0, 1))
        mean_sum += individual_mean
        std_sum += individual_stdev

    mean = mean_sum / num_images
    std = std_sum / num_images
    return mean.astype(np.float32), std.astype(np.float32)