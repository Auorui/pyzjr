from math import ceil
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

import pyzjr.Z as Z
from .io import StackedImages

IMAGENET_MEAN = Z.IMAGENET_MEAN
IMAGENET_STD  = Z.IMAGENET_STD

def normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
    # 按BCHW格式的ImageNet统计数据对RGB图像x进行规范化, i.e. = (x - mean) / std
    return TF.normalize(x, mean, std, inplace=inplace)

def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    # 按BCHW格式的ImageNet统计数据对RGB图像x进行反规范化, i.e. = x * std + mean
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x

def augmentHsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV颜色空间增强
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed

def hist_equalize(im, clahe=True, bgr=False):
    # 均衡BGR图像“im”上的直方图，其形状为im(n，m，3)，范围为0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)


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

class ToTensor:
    def __init__(self, half=False):
        super().__init__()
        self.half = half

    def __call__(self, img):  # im = np.array HWC in BGR order
        img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        img = torch.from_numpy(img)  # to torch
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0-255 to 0.0-1.0
        return img

class ToNumpy(object):
    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img,

class CenterCrop:
    # T.Compose([CenterCrop(size), ToTensor()])
    def __init__(self, size=640):
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top:top + m, left:left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)

def mixup(img1, labels, img2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return img, labels

def base_crop(img, x_min, y_min, x_max, y_max):
    height, width = img.shape[:2]
    assert x_max > x_min, "[pyzjr]:Maximum value of cropping x_max cannot be less than the minimum value x_min"
    assert y_max > y_min, "[pyzjr]:Maximum value of cropping y_max cannot be less than the minimum value y_min"
    assert x_min >= 0, "[pyzjr]:x_min cannot be less than 0"
    assert x_max <= width, "[pyzjr]:x_max cannot be greater than the image width"
    assert y_min >= 0, "[pyzjr]:y_min cannot be less than 0"
    assert y_max <= height, "[pyzjr]:y_max cannot be greater than the image height"

    return img[y_min:y_max, x_min:x_max]

def base_crop_block(height, width, crop_height, crop_width, h_start, w_start):
    """
    :param height:图像原始高度
    :param width:图像原始宽度
    :param crop_height:裁剪高度
    :param crop_width:裁剪宽度
    :param h_start:[0, 1),有效起始位置中的一个随机选择
    :param w_start:[0, 1),同上
    """
    # h_start is [0, 1) and should map to [0, (height - crop_height)]  (note inclusive)
    # This is conceptually equivalent to mapping onto `range(0, (height - crop_height + 1))`
    y1 = int((height - crop_height + 1) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width + 1) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2

def flip(image, option_value):
    """
    :param image : numpy array of image
    :param option_value: random integer between 0 to 3
            vertical                          0
            horizontal                        1
            horizontally and vertically flip  2
    Return: image : numpy array of flipped image
    """
    if option_value == 0:
        image = np.flip(image, option_value)
    elif option_value == 1:
        image = np.flip(image, option_value)
    elif option_value == 2:
        image = np.flip(image, 0)
        image = np.flip(image, 1)
    else:
        image = image

    return image

def brightness(img,brightness_factor = 1.5):
    """
    图像增广——图像亮度调整
    :param img: 输入图像
    :param brightness_factor: 亮度调整因子，默认为1.5
    :return: 返回亮度调整后的图像
    """
    image_float = img.astype(np.float32)
    adjusted_image = image_float * brightness_factor
    # 将图像像素值限制在[0, 255]范围内
    adjusted_image = np.clip(adjusted_image, 0, 255)
    adjusted_image = adjusted_image.astype(np.uint8)
    return adjusted_image

def Centerzoom(img, zoom_factor: int):
    """中心缩放"""
    h, w = img.shape[:2]
    h_ch, w_ch = ceil(h / zoom_factor), ceil(w / zoom_factor)
    h_top, w_top = (h - h_ch) // 2, (w - w_ch) // 2
    zoomed_img = cv2.resize(img[h_top : h_top + h_ch, w_top : w_top + w_ch], (w, h), interpolation=cv2.INTER_LINEAR)

    return zoomed_img


def Stitcher_image(image_paths):
    """
    图像增广——图像拼接，图片较小可能拼接失败
    :param image_paths: 由图片路径组成的列表
    :return: 返回被拼接好的图片
    """
    stitcher = cv2.Stitcher.create()  # opencv4可以用这个,opencv3的可以使用cv2.createStitcher
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    assert len(images) >= 2, "[pyzjr]:At least two images are required for stitching"
    (status, stitched_image) = stitcher.stitch(images)
    assert status == cv2.Stitcher_OK, '[pyzjr]:Image stitching failed'
    return stitched_image

def BilinearImg(image, scale):
    """
    双线性插值 https://blog.csdn.net/m0_62919535/article/details/132094815
    :param image: 原始图像。
    :param scale: 规格,可使用小数
    """
    ah, aw, channel = image.shape
    bh, bw = int(ah * scale), int(aw * scale)
    dst_img = np.zeros((bh, bw, channel), np.uint8)

    y_coords, x_coords = np.meshgrid(np.arange(bh), np.arange(bw), indexing='ij')
    AX = (x_coords + 0.5) / scale - 0.5
    AY = (y_coords + 0.5) / scale - 0.5

    x1 = np.floor(AX).astype(int)
    y1 = np.floor(AY).astype(int)
    x2 = np.minimum(x1 + 1, aw - 1)
    y2 = np.minimum(y1 + 1, ah - 1)
    R1 = ((x2 - AX)[:, :, np.newaxis] * image[y1, x1]).astype(float) + (
            (AX - x1)[:, :, np.newaxis] * image[y1, x2]).astype(float)
    R2 = ((x2 - AX)[:, :, np.newaxis] * image[y2, x1]).astype(float) + (
            (AX - x1)[:, :, np.newaxis] * image[y2, x2]).astype(float)

    dst_img = (y2 - AY)[:, :, np.newaxis] * R1 + (AY - y1)[:, :, np.newaxis] * R2

    return dst_img.astype(np.uint8)

def blur(img, ksize: int):
    """均值滤波 """
    blur_img = cv2.blur(img, ksize=(ksize, ksize))
    return blur_img

def median_blur(img, ksize: int):
    """中值滤波"""
    if img.dtype == np.float32 and ksize not in {3, 5, 7}:
        raise ValueError(f"Invalid ksize value {ksize}.The available values are 3, 5, and 7")
    medblur_img = cv2.medianBlur(img, ksize=ksize)
    return medblur_img

def gaussian_blur(img, ksize: int):
    """
    高斯模糊,提供给不熟悉高斯模糊参数的用户,sigma根据ksize进行自动计算,具体可以参考下面
    https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
    """
    sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    gaussianblur_img = cv2.GaussianBlur(img, ksize=(ksize, ksize), sigmaX=sigma)
    return gaussianblur_img

def bilateral_filter(img, d=10, sigmaColor=10, sigmaSpace=10, showTrack=True):
    """
    双边滤波，添加了轨迹栏的功能
    :param img: 输入图片
    :param showTrack: 是否打开轨迹栏,获得调试参数
    :param d: 决定了双边滤波器在像素范围内的大小。较大的值将考虑更多的像素，但可能导致模糊效果较强烈。建议的范围是5到50之间。你可以从5开始尝试，然后根据需要逐渐增加。
    :param sigmaColor:表示颜色空间中的标准差，影响滤波器在颜色空间中的平滑程度。较大的值将允许更多的颜色变化，但也可能导致图像过于平滑。建议的范围是10到100之间
    :param sigmaSpace:表示空间坐标中的标准差，控制滤波器在像素空间内的平滑程度。较大的值将允许更多的像素变化，但可能导致图像过于平滑。建议的范围也是10到100之间。
    :return:
    """
    def empty(a):
        pass
    if showTrack:
        cv2.namedWindow('image')
        cv2.createTrackbar('d', 'image', 1, 50, empty)
        cv2.createTrackbar('sigmaColor', 'image', 1, 150, empty)
        cv2.createTrackbar('sigmaSpace', 'image', 1, 150, empty)
        while True:
            d = cv2.getTrackbarPos('d', 'image')
            sigmaColor = cv2.getTrackbarPos('sigmaColor', 'image')
            sigmaSpace = cv2.getTrackbarPos('sigmaSpace', 'image')
            dst = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
            stackimg = StackedImages(0.5, [img, dst])
            cv2.imshow('image', stackimg)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k == ord("y"):
                print(d, sigmaColor, sigmaSpace)
    else:
        dst = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
        return dst

class Retinex():
    """
    增强算法Retinex
    - 单尺度 : SSR
    - 多尺度 : MSR
    - 多尺度自适应增益 : MSRCR
    https://blog.csdn.net/m0_62919535/article/details/130372571
    """
    def SSR(self, img, sigma):
        """
        将输入图像转换为了对数空间。/255将像素值归一化到0到1之间，np.log1p取对数并加1是为了避免出现对数运算中分母为0的情况。二维离散傅里叶变换将
        图像从空间域变换到频率域，可以提取出图像中的频率信息。G_recs是用于计算高斯核的半径，result用于最后三通道的叠加。然后循环用于计算加权后的
        频率域图像，再逆二维离散傅里叶变换，得到反射图像，对反射图像进行指数变换，得到最终的输出图像。
        :param img: 输入图像
        :param sigma: 高斯分布的标准差
        :return:
        """
        img_log = np.log1p(np.array(img, dtype="float") / 255)
        img_fft = np.fft.fft2(img_log)
        G_recs = sigma // 2 + 1
        result = np.zeros_like(img_fft)
        rows, cols, deep = img_fft.shape
        for z in range(deep):
            for i in range(rows):
                for j in range(cols):
                    for k in range(1, G_recs):
                        G = np.exp(-((np.log(k) - np.log(sigma)) ** 2) / (2 * np.log(2) ** 2))
                        #计算高斯滤波器的权值，其中sigma是高斯分布的标准差，k是高斯滤波器的半径，G是高斯滤波器在该点的权值。
                        result[i, j] += G * img_fft[i, j]
        img_ssr = np.real(np.fft.ifft2(result))
        img_ssr = np.exp(img_ssr) - 1
        img_ssr = np.uint8(cv2.normalize(img_ssr, None, 0, 255, cv2.NORM_MINMAX))
        return img_ssr

    def MSR(self, img, scales):
        """
        MSR算法在图像增强中与SSR不同的是，它不需要进行频域变换，它主要是基于图像在多个尺度下的平滑处理和差分处理来提取图像的局部对比度信息和全
        局对比度信息，从而实现对图像的增强。
        在 MSR 算法中，先对图像进行对数变换得到对数图像，然后在不同的尺度下，使用高斯滤波对图像进行平滑处理，得到不同尺度下的平滑图像。接着，通
        过将对数图像和不同尺度下的平滑图像进行差分，得到多个尺度下的细节图像。最后，将这些细节图像加权融合，输出最终的增强图像。

        :param img:
        :param scales: 取值大概在1-10之间
        :return:
        """
        img_log = np.log1p(np.array(img, dtype="float") / 255)
        result = np.zeros_like(img_log)
        img_light = np.zeros_like(img_log)
        r, c, deep = img_log.shape
        for z in range(deep):
            for scale in scales:
                kernel_size = scale * 4 + 1
                # 高斯滤波器的大小，经验公式kernel_size = scale * 4 + cat
                sigma = scale
                img_smooth = cv2.GaussianBlur(img_log[:, :, z], (kernel_size, kernel_size), sigma)
                img_detail = img_log[:, :, z] - img_smooth
                result[:, :, z] += cv2.resize(img_detail, (c, r))
                img_light[:, :, z] += cv2.resize(img_smooth, (c, r))
        img_msr = np.exp(result+img_light) - 1
        img_msr = np.uint8(cv2.normalize(img_msr, None, 0, 255, cv2.NORM_MINMAX))
        return img_msr


    def MSRCR(self, img, scales, k):
        """

        :param img:
        :param scales:取值大概在1-10之间
        :param k: k的取值范围在10~20之间比较合适。当k取值较小时，图像的细节增强效果比较明显，但会出现较强的噪点，当k取值较大时，图像的细节
                    增强效果不明显，但噪点会减少。
        :return:
        """
        img_log = np.log1p(np.array(img, dtype="float") / 255)
        result = np.zeros_like(img_log)
        img_light = np.zeros_like(img_log)
        r, c, deep = img_log.shape
        for z in range(deep):
            for scale in scales:
                kernel_size = scale * 4 + 1
                # 高斯滤波器的大小，经验公式kernel_size = scale * 4 + cat
                sigma = scale
                G_ratio=sigma**2/(sigma**2+k)
                img_smooth = cv2.GaussianBlur(img_log[:, :, z], (kernel_size, kernel_size), sigma)
                img_detail = img_log[:, :, z] - img_smooth
                result[:, :, z] += cv2.resize(img_detail, (c, r))
                result[:, :, z]=result[:, :, z]*G_ratio
                img_light[:, :, z] += cv2.resize(img_smooth, (c, r))

        img_msrcr = np.exp(result+img_light) - 1
        img_msrcr = np.uint8(cv2.normalize(img_msrcr, None, 0, 255, cv2.NORM_MINMAX))
        return img_msrcr

class Filter():
    """手写实现,仅供学习参考,最好还是使用cv2
    https://blog.csdn.net/m0_62919535/category_11936595.html?spm=1001.2014.3001.5482
    """
    def median_filtering(self, img,ksize=3):
        """
        中值滤波
        :param img:输入图像
        :param ksize: 核大小
        :return: 中值滤波平滑
        """
        h, w, c = img.shape
        half = ksize//2
        dst = np.zeros((h+2*half,w+2*half,c),np.uint8)
        dst[half:half+h, half:half+w] = img.copy()

        tmp=dst.copy()
        for y in range(h):
            for x in range(w):
                for z in range(c):
                    dst[half+x,half+y]=np.median(tmp[x:x+ksize,y:y+ksize])
        output=dst[half:half+h,half:half+w]
        return output

    def Arerage_Filtering(self, img, k_size=3):
        """
        均值滤波函数,默认会返回灰度图，因为三个for循环实在是太耗费时间了。而且，在这里需要考虑到边界点的问题。
        计算填充的宽度，即卷积核宽度的一半，用于处理图像边缘。使用cv2.copyMakeBorder函数进行边缘填充，将图
        像的边缘复制并填充到周围，以防止边缘像素点无法进行卷积。
        :param img: 原始图像，要求为灰度图
        :param k_size: 滤波核大小,默认为3,确保滤波核大小为奇数
        :return: 处理后的均值滤波图像
        """
        if k_size % 2 == 0:
            k_size += 1
        rows, cols = img.shape[:2]
        # 计算需要在图像边界扩充的大小
        pad_width = (k_size - 1) // 2
        # 在图像边界进行扩充
        img_pad = cv2.copyMakeBorder(img, pad_width, pad_width, pad_width, pad_width, cv2.BORDER_REPLICATE)
        img_filter = np.zeros_like(img)
        for i in range(rows):
            for j in range(cols):
                pixel_values = img_pad[i:i+k_size, j:j+k_size].flatten()
                img_filter[i, j] = np.mean(pixel_values)

        return img_filter

    def gaussian_kernel(self, size, sigma):
        """
        生成高斯核
        :param size: 核的大小
        :param sigma: 标准差
        :return:
        """
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2)/(2*sigma**2))
        kernel /= 2 * np.pi * sigma**2
        kernel /= np.sum(kernel)
        return kernel

    def Gaussian_Filtering(self, img, kernel_size, sigma):
        """
        生成高斯滤波
        :param img: 输入图像
        :param kernel_size: 核大小
        :param sigma: 标准差
        :return:
        """
        kernel = self.gaussian_kernel(kernel_size, sigma)
        height, width, _ = img.shape
        result = np.zeros_like(img, dtype=np.float32)

        pad_size = kernel_size // 2
        img_pad = np.pad(img, [(pad_size, pad_size), (pad_size, pad_size), (0, 0)], mode='constant')
        for c in range(_):
            for i in range(pad_size, height + pad_size):
                for j in range(pad_size, width + pad_size):
                    result[i - pad_size, j - pad_size, c] = np.sum(kernel * img_pad[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1, c])
        return np.uint8(result)

