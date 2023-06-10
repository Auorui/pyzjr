"""
-图形处理相关
-滤波算法
    类Filter()
    -中值滤波 ： median_filtering
    -双边滤波 ： Bilateral_filtering
    -均值滤波 ： Arerage_Filtering
    -高斯滤波 ： Gaussian_Filtering
-增广算法
    类Enhance()
    -旋转 ： Rotated_image
    -亮度调整 ： Adjusted_image
    -裁剪 ： Cut_image
    -拼接 ： Stitcher_image
    类Random_Enhance()
    -垂直或水平翻转 ： horizontal_flip
    -随机参数 ： random_generate
    -随机翻转 ： random_flip_batch
    -随机明暗调整 ： random_brightness_batch
    -随机裁剪 : random_Cropping_batch
"""
import numpy as np
import cv2
import pyzjr.utils as zjr
from pyzjr.utils import empty
import random

class Filter():
    def median_filtering(self, img,ksize=3):
        """
        中值滤波
        :param img:输入图像
        :param ksize: 核大小
        :return: 中值滤波平滑
        """
        h,w,c=img.shape
        half=ksize//2
        dst=np.zeros((h+2*half,w+2*half,c),np.uint8)
        dst[half:half+h,half:half+w]=img.copy()

        tmp=dst.copy()
        for y in range(h):
            for x in range(w):
                for z in range(c):
                    dst[half+x,half+y]=np.median(tmp[x:x+ksize,y:y+ksize])
        output=dst[half:half+h,half:half+w]
        return output

    def Bilateral_filtering(self, img,showTrack=True,d=10, sigmaColor=10, sigmaSpace=10):
        """
        双边滤波，添加了轨迹栏的功能
        :param img: 输入图片
        :param showTrack:
        :param d:
        :param sigmaColor:
        :param sigmaSpace:
        :return:
        """
        if showTrack:
            cv2.namedWindow('image')
            cv2.createTrackbar('d', 'image', 1, 50, empty)
            cv2.createTrackbar('sigmaColor', 'image', 1, 150, empty)
            cv2.createTrackbar('sigmaSpace', 'image', 1, 150, empty)
            while True:
                d = cv2.getTrackbarPos('d', 'image')
                sigmaColor = cv2.getTrackbarPos('sigmaColor', 'image')
                sigmaSpace = cv2.getTrackbarPos('sigmaSpace', 'image')
                print(d, sigmaColor, sigmaSpace)
                dst = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
                stackimg = zjr.stackImages(0.5, [img, dst])
                cv2.imshow('image', stackimg)

                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
        else:
            dst = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
            cv2.imshow('image', dst)
            cv2.waitKey(0) & 0xFF

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

class Enhance():
    def __init__(self,image):
        """
        初始化全局参数
        :param image: 输入图像
        """
        self.img=image.copy()
        self.height, self.width = self.img.shape[:2]
        self.center = (self.width // 2, self.height // 2)

    def Rotated_image(self,img, angle = 45, scale = 1.0):
        """
        图像增广——图像旋转
        :param img: 输入图像
        :param angle: 旋转角度
        :param scale: 缩放比例
        :return: 返回旋转后的图像
        """
        matrix = cv2.getRotationMatrix2D(self.center, angle, scale) #旋转中心，旋转角度，缩放比例
        rotated_image = cv2.warpAffine(img, matrix, (self.width, self.height))
        return rotated_image

    def Adjusted_image(self,img,brightness_factor = 1.5):
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

    def Cut_image(self,image, coordinate, Leath, cropImg=False, bgColor=128, save=False, saveFile=''):
        """
        图像裁剪——裁剪图像并用灰色填充不足的部分，进行边界检测
        :param image: 输入图像
        :param coordinate: 列表，左上角坐标
        :param Leath: 列表，裁剪所需的长宽度
        :param cropImg: 是否保存裁剪的图片，
        :param save: 保存图片，没有灰度条，默认为False
        :param saveFile: 保存路径，默认为空
        :return: 返回裁剪后的图像
        """
        imageH, imageW = image.shape[:2]
        x, y = coordinate[0], coordinate[1]
        width, height = Leath[0], Leath[1]
        h, w = height, width
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + width > imageW:
            width = imageW - x
        if y + height > imageH:
            height = imageH - y

        cropped_image = image[y:y + height, x:x + width]
        padded_image = np.full((h, w, 3), bgColor, dtype=np.uint8)
        x_offset = (w - width) // 2
        y_offset = (h - height) // 2
        padded_image[y_offset:y_offset + height, x_offset:x_offset + width] = cropped_image
        if save:
            cv2.imwrite(saveFile, cropped_image)
        if cropImg:
            return cropped_image
        else:
            return padded_image

    def Stitcher_image(self,image_paths):
        """
        图像增广——图像拼接，图片较小可能拼接失败
        :param image_paths: 由图片路径组成的列表
        :return: 返回被拼接好的图片
        """
        stitcher = cv2.Stitcher_create()
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
        if len(images) < 2:
            print('至少需要两个图像进行拼接')
            return
        (status, stitched_image) = stitcher.stitch(images)
        if status == cv2.Stitcher_OK:
            return stitched_image
        else:
            print('图像拼接失败')


class Random_Enhance():
    def __init__(self,img):
        self.flip_values=[0, 1, -1, 2]
        self.bright_values=[0.5, 1.5]
        self.cut_scale_values=[0.1, 1.0]
        self.img=img.copy()
        self.en=Enhance(img)
        self.h, self.w = self.img.shape[:2]

    def horizontal_flip(self,image, axis):
        """
        对图像进行垂直或水平翻转，可以组合
        :param image: 输入图像
        :param axis: 各自以25 % 的可能性。
                    0 : 垂直翻转，
                    1 : 水平翻转，
                   -1 : 水平垂直翻转，
                    2 : 不翻转，
        :return:
        """
        if axis != 2:
            image = cv2.flip(image, axis)
        return image

    def random_generate(self, mode):
        if mode == 'flip':
            randimg = np.random.choice(self.flip_values)
            return randimg
        elif mode == 'bright':
            val0, val1 = self.bright_values
            bri_num = round(random.uniform(val0, val1), 1)
            return bri_num
        elif mode == 'cut':
            val0, val1 = self.cut_scale_values
            scale = round(random.uniform(val0, val1), 1)
            newH = random.randint(0, self.h)
            newW = random.randint(0, self.w)
            initPoint = [newW, newH]
            return initPoint, scale
        else:
            print("这并不是在选定规格内 This is not within the selected specifications.")


    def random_flip_batch(self,images):
        """
        随机翻转
        :param images: 输入图像构成的列表
        :return:
        """
        imglist = []
        for img in images:
            random_value = self.random_generate("flip")
            image = self.horizontal_flip(img, random_value)
            imglist.append(image)
        return imglist

    def random_brightness_batch(self,images):
        """
        随机明暗调整
        :param images: 输入图像构成的列表
        :return:
        """
        imglist = []
        for img in images:
            random_value = self.random_generate("bright")
            image = self.en.Adjusted_image(img, random_value)
            imglist.append(image)
        return imglist

    def random_Cropping_batch(self,images):
        """
        随机裁剪
        :param images: 输入图像构成的列表
        :return:
        """
        imglist = []
        for img in images:
            InitPoint, scale = self.random_generate("cut")
            newH, newW = self.h * scale, self.w * scale
            image = self.en.Cut_image(img, InitPoint, [int(newH), int(newW)], cropImg=True)
            restored_image = cv2.resize(image, (self.w, self.h))
            imglist.append(restored_image)
        return imglist

