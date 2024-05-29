import cv2
import numpy as np

def meanblur(img, ksize):
    """均值滤波 """
    blur_img = cv2.blur(img, ksize=ksize)
    return blur_img

def medianblur(img, ksize):
    """中值滤波"""
    if img.dtype == np.float32 and ksize not in {3, 5, 7, 9, 11}:
        raise ValueError(f"Invalid ksize value {ksize}.The available values are 3, 5, and 7")
    medblur_img = cv2.medianBlur(img, ksize=ksize)
    return medblur_img

def gaussianblur(img, ksize):
    """
    高斯模糊, 提供给不熟悉高斯模糊参数的用户, sigma根据ksize进行自动计算,具体可以参考下面
    https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
    """
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    gaussianblur_img = cv2.GaussianBlur(img, ksize=(ksize, ksize), sigmaX=sigma)
    return gaussianblur_img

def bilateralblur(img, d=5, sigma_color=75, sigma_space=75):
    """双边滤波"""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

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