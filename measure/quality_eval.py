import cv2
import torch
import numpy as np
import warnings
from functools import partial
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pyzjr.utils.check import is_numpy, is_tensor, is_gray_image
from math import exp


def gradient_metric(image, size=11, edge_pixels=1, grad_type='sobel', filter_type='blur', metric_type='max'):
    """计算指示图像中模糊强度的度量, 越接近 0 越清晰, 越接近 1 越模糊
    cv2.CV_32F : 5
    :param image: 应为灰度图
    :param size: 滤波器的大小
    :param edge_pixels: 从图像边缘裁剪的像素数量，以减少边界效应
    :param grad_type: 选择使用的梯度算子，'sobel' 或 'scharr'
    :param filter_type: 选择使用的滤波器类型，'blur', 'gaussian', 'median'
    :param metric_type: 选择返回“最大”还是“平均”的模糊度量值，'max' 或 'mean'
    :return: 图像模糊强度的度量值, 属于 (0, 1)
    """
    grad_dict = {
        'sobel': partial(cv2.Sobel, ksize=3),
        'scharr': cv2.Scharr
    }
    filter_dict = {
        'blur': partial(cv2.blur, ksize=(size, size)),
        'gaussian': partial(cv2.GaussianBlur, ksize=(size, size), sigmaX=0.3 * ((size - 1) * 0.5 - 1) + 0.8),
        'median': partial(cv2.medianBlur, ksize=size)
    }
    if grad_type not in grad_dict:
        raise ValueError(f"Invalid grad_type '{grad_type}', must be 'sobel' or 'scharr'.")
    if filter_type not in filter_dict:
        raise ValueError(f"Invalid filter_type '{filter_type}', must be 'blur', 'gaussian', or 'median'.")
    if metric_type not in ['max', 'mean']:
        raise ValueError(f"Invalid metric_type '{metric_type}', must be 'max' or 'mean'.")
    grad_operator = grad_dict[grad_type]
    filter_operator = filter_dict[filter_type]
    if not is_gray_image(image):
        warnings.warn("The input image is not a grayscale image, it will be converted to grayscale.", UserWarning)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_metric = []
    h, w = image.shape[:2]
    image = image[edge_pixels:h - edge_pixels, edge_pixels:w - edge_pixels]
    blurred_image = filter_operator(image)

    for ax in range(image.ndim):  # x, y 方向
        original_gradient = np.abs(grad_operator(image, 5, ax, 1 - ax))
        blurred_gradient = np.abs(grad_operator(blurred_image, 5, ax, 1 - ax))
        g_differ = np.maximum(original_gradient - blurred_gradient, 0)
        g1 = np.sum(original_gradient)
        g2 = np.sum(g_differ)
        results = np.clip((g1 - g2) / g1, 0, 1)
        blur_metric.append(results)
    if metric_type == 'max':
        return np.max(blur_metric)
    elif metric_type == 'mean':
        return np.mean(blur_metric)

def calculate_psnrV1(input_, target_):
    """计算两张图片的PSNR"""
    img1 = np.array(input_)
    img2 = np.array(target_)
    psnr = peak_signal_noise_ratio(img1, img2)
    return psnr

def calculate_psnrV2(input_, target_):
    psnr = 0
    if is_numpy(input_) and is_numpy(target_):
        mse = np.mean((input_ - target_) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1. if input_.max() <= 1 else 255.
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
    elif is_tensor(input_) and is_tensor(target_):
        psnr = 10 * torch.log10(1 / F.mse_loss(input_, target_)).item()
    return psnr

def calculate_ssimV1(input_, target_, win_size=7):
    """计算两张图片的 SSIM"""
    input_img = np.array(input_)
    target_img = np.array(target_)
    height, width = input_img.shape[:2]
    win_size = min(win_size, min(height, width))
    win_size = win_size + 1 if win_size % 2 == 0 else win_size
    ssim_value = structural_similarity(input_img, target_img, win_size=win_size, channel_axis=-1)
    return ssim_value

def calculate_ssimV2(input_, target_, ksize=11, sigma=1.5):
    ssim_map = 0
    if is_numpy(input_) and is_numpy(target_):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = input_.astype(np.float64)
        img2 = target_.astype(np.float64)
        kernel = cv2.getGaussianKernel(ksize, sigma)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))

    if is_tensor(input_) and is_tensor(target_):
        (_, channel, _, _) = input_.size()
        gauss = torch.Tensor([exp(-(x - ksize/2)**2/float(2 * sigma ** 2)) for x in range(ksize)])
        gauss = gauss / gauss.sum()
        _1D_window = gauss.unsqueeze(1).to(input_.device)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, ksize, ksize)
        mu1 = F.conv2d(input_, window, padding = ksize//2, groups = channel)
        mu2 = F.conv2d(target_, window, padding = ksize//2, groups = channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        sigma1_sq = F.conv2d(input_*input_, window, padding=ksize//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target_*target_, window, padding=ksize//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(input_*target_, window, padding=ksize//2, groups=channel) - mu1_mu2
        C1 = 0.01**2
        C2 = 0.03**2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


if __name__=="__main__":
    img1 = cv2.imread('1.png')  # input
    img2 = cv2.imread('1_1.png')  # target

    if img1.shape != img2.shape:
        print("Images must have the same dimensions.")
        exit()

    psnr_score = calculate_psnrV1(img1, img2)
    print(f'PSNR: {psnr_score:.2f} dB')
    psnr_score = calculate_psnrV2(img1, img2)
    print(f'PSNR V2: {psnr_score:.2f} dB')

    ssim_score = calculate_ssimV1(img1, img2)
    print(f'SSIM: {ssim_score:.4f}')
    ssim_score = calculate_ssimV2(img1, img2)
    print(f'SSIM V2: {ssim_score:.4f}')
    img1_tensor = torch.tensor(np.transpose(img1 / 255.0, (2, 0, 1)), dtype=torch.float32).unsqueeze(0)
    img2_tensor = torch.tensor(np.transpose(img2 / 255.0, (2, 0, 1)), dtype=torch.float32).unsqueeze(0)
    # print(img1_tensor.shape, img2_tensor.shape)
    psnr = calculate_psnrV2(img1_tensor, img2_tensor)
    print(f'PSNR V2 Tensor: {psnr:.2f} dB')
    ssim = calculate_ssimV2(img1_tensor, img2_tensor)
    print(f'SSIM V2 Tensor: {ssim:.4f}')