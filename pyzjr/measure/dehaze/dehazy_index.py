import cv2
import torch
import numpy as np
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pyzjr.core.general import is_numpy, is_tensor

def calculate_psnr(input_, target_):
    """计算两张图片的PSNR"""
    img1 = np.array(input_)
    img2 = np.array(target_)
    psnr = peak_signal_noise_ratio(img1, img2)
    return psnr

def calculate_psnrv2(input_, target_):
    psnr = 0
    if is_numpy(input_) and is_numpy(target_):
        mse = np.mean((input_ - target_) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = max(np.max(input_), np.max(target_))
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
    elif is_tensor(input_) and is_tensor(target_):
        psnr = 10 * torch.log10(1 / F.mse_loss(input_, target_)).item()
    return psnr

def calculate_ssim(input_, target_, win_size=7):
    """计算两张图片的 SSIM"""
    input_img = np.array(input_)
    target_img = np.array(target_)
    height, width = input_img.shape[:2]
    win_size = min(win_size, min(height, width))
    win_size = win_size + 1 if win_size % 2 == 0 else win_size
    ssim_value = structural_similarity(input_img, target_img, win_size=win_size, channel_axis=-1)
    return ssim_value

def calculate_ssimv2(input_, target_, ksize=11, sigma=1.5):
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
    return ssim_map.mean()

if __name__=="__main__":
    img1 = cv2.imread('221.png')
    img2 = cv2.imread('220.png')

    if img1.shape != img2.shape:
        print("Images must have the same dimensions.")
        exit()


    psnr = calculate_psnr(img1, img2)
    print(f'PSNR: {psnr:.2f} dB')
    psnr = calculate_psnrv2(img1, img2)
    print(f'PSNR V2: {psnr:.2f} dB')

    ssim_score = calculate_ssim(img1, img2)
    print(f'SSIM: {ssim_score:.4f}')
    ssim_score = calculate_ssimv2(img1, img2)
    print(f'SSIM V2: {ssim_score:.4f}')
