import cv2
import torch
import warnings
import numpy as np
from functools import partial
import torch.nn.functional as F


def calculate_psnr(inputs, targets):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    Supports both NumPy arrays (BGR/RGB, uint8/float) and PyTorch Tensors (BCHW).

    Args:
        inputs (np.ndarray or torch.Tensor): The input/predicted image.
        targets (np.ndarray or torch.Tensor): The target/reference image.

    Returns:
        float: The PSNR value in dB. Returns float('inf') if the images are identical.
    """
    psnr = 0.0
    if isinstance(inputs, np.ndarray) and isinstance(targets, np.ndarray):
        input_ = inputs.astype(np.float64)
        target_ = targets.astype(np.float64)
        mse = np.mean((input_ - target_) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1. if input_.max() <= 1 else 255.
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
    elif torch.is_tensor(inputs) and torch.is_tensor(targets):
        psnr = 10 * torch.log10(1 / F.mse_loss(inputs, targets)).item()
    return psnr


def calculate_ssim(inputs, targets, ksize=11, sigma=1.5):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    Supports both NumPy arrays (HWC) and PyTorch Tensors (CHW or BCHW).

    Args:
        inputs (np.ndarray or torch.Tensor): The input/predicted image.
        targets (np.ndarray or torch.Tensor): The target/reference image.
        ksize (int): The size of the Gaussian kernel. Defaults to 11.
        sigma (float): The standard deviation of the Gaussian kernel. Defaults to 1.5.

    Returns:
        float: The SSIM value.
    """
    ksize = ksize + 1 if ksize % 2 == 0 else ksize
    pad = ksize // 2

    if isinstance(inputs, np.ndarray) and isinstance(targets, np.ndarray):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = inputs.astype(np.float64)
        img2 = targets.astype(np.float64)
        kernel = cv2.getGaussianKernel(ksize, sigma)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[pad:-pad, pad:-pad]
        mu2 = cv2.filter2D(img2, -1, window)[pad:-pad, pad:-pad]

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[pad:-pad, pad:-pad] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[pad:-pad, pad:-pad] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[pad:-pad, pad:-pad] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    elif torch.is_tensor(inputs) and torch.is_tensor(targets):
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
            targets = targets.unsqueeze(0)

        _, channel, _, _ = inputs.size()
        gauss = torch.Tensor([np.exp(-(x - ksize / 2) ** 2 / float(2 * sigma ** 2)) for x in range(ksize)])
        gauss = gauss / gauss.sum()
        _1D_window = gauss.unsqueeze(1).to(inputs.device)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, ksize, ksize).to(inputs.device)

        mu1 = F.conv2d(inputs, window, padding=pad, groups=channel)
        mu2 = F.conv2d(targets, window, padding=pad, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(inputs * inputs, window, padding=pad, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(targets * targets, window, padding=pad, groups=channel) - mu2_sq
        sigma12 = F.conv2d(inputs * targets, window, padding=pad, groups=channel) - mu1_mu2

        sigma1_sq = torch.clamp(sigma1_sq, min=0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean().item()

    else:
        raise TypeError("Both input_ and target_ must be either NumPy arrays or PyTorch Tensors.")


def calculate_blur(image, size=11, edge_pixels=1, grad_type='sobel', filter_type='blur', metric_type='max'):
    """
    Calculate the blur metric based on gradient variation.
    A value closer to 0 means sharper, closer to 1 means blurrier.

    Args:
        image (np.ndarray): Input image (will be converted to grayscale if not already).
        size (int): Kernel size for the blur filter. Defaults to 11.
        edge_pixels (int): Number of pixels to crop from edges to avoid boundary effects. Defaults to 1.
        grad_type (str): Gradient operator, either 'sobel' or 'scharr'. Defaults to 'sobel'.
        filter_type (str): Blur filter type ('blur', 'gaussian', 'median'). Defaults to 'blur'.
        metric_type (str): Aggregation method, either 'max' or 'mean'. Defaults to 'max'.

    Returns:
        float: Blur metric value in range [0, 1].
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

    # Safely convert to grayscale
    is_gray = (image.ndim == 2) or (image.ndim == 3 and image.shape[2] == 1)

    if not is_gray:
        warnings.warn("The input image is not grayscale, converting...", UserWarning)
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape[:2]
    # Avoid edge effects
    image = image[edge_pixels:h - edge_pixels, edge_pixels:w - edge_pixels]
    blurred_image = filter_operator(image)

    blur_metrics = []
    for ax in [0, 1]:
        original_gradient = np.abs(grad_operator(image, 5, ax, 1 - ax))
        blurred_gradient = np.abs(grad_operator(blurred_image, 5, ax, 1 - ax))

        g_differ = np.maximum(original_gradient - blurred_gradient, 0)
        g1 = np.sum(original_gradient)
        g2 = np.sum(g_differ)

        if g1 == 0:
            results = 0.0
        else:
            results = np.clip((g1 - g2) / g1, 0, 1)
        blur_metrics.append(results)

    return np.max(blur_metrics) if metric_type == 'max' else np.mean(blur_metrics)


if __name__=="__main__":
    img1 = cv2.imread(r'/resources/images/ship_hazy.png')  # input
    img2 = cv2.imread(r'/resources/images/ship_gt.png')  # target

    blur_image = cv2.GaussianBlur(img2, ksize=(9, 9), sigmaX=3, sigmaY=0)
    blur_score = calculate_blur(blur_image)
    print(blur_score)


    if img1.shape != img2.shape:
        print("Images must have the same dimensions.")
        exit()

    psnr_score = calculate_psnr(img1, img2)
    print(f'PSNR: {psnr_score:.2f} dB')
    ssim_score = calculate_ssim(img1, img2)
    print(f'SSIM: {ssim_score:.4f}')

    img1_tensor = torch.tensor(np.transpose(img1 / 255.0, (2, 0, 1)), dtype=torch.float32).unsqueeze(0)
    img2_tensor = torch.tensor(np.transpose(img2 / 255.0, (2, 0, 1)), dtype=torch.float32).unsqueeze(0)
    # print(img1_tensor.shape, img2_tensor.shape)
    psnr = calculate_psnr(img1_tensor, img2_tensor)
    print(f'PSNR Tensor: {psnr:.2f} dB')
    ssim = calculate_ssim(img1_tensor, img2_tensor)
    print(f'SSIM Tensor: {ssim:.4f}')