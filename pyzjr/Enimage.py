"""
- author:Auorui(夏天是冰红茶)
- creation time:2022.10.11
- blog : https://blog.csdn.net/m0_62919535/category_11936595.html?spm=1001.2014.3001.5482
- 图形处理相关
- 滤波算法
    类Filter()
    - 中值滤波 : median_filtering
    - 双边滤波 : Bilateral_filtering
    - 均值滤波 : Arerage_Filtering
    - 高斯滤波 : Gaussian_Filtering
-增广算法
    类Enhance()
    - 旋转 : Rotated_image
    - 亮度调整 : Adjusted_image
    - 裁剪 : Cut_image
    - 拼接 : Stitcher_image
    类Random_Enhance()
    - 垂直或水平翻转 : horizontal_flip
    - 随机参数 : random_generate
    - 随机翻转 : random_flip_batch
    - 随机明暗调整 : random_brightness_batch
    - 随机裁剪 : random_Cropping_batch
-增强算法
    类Retinex()
    - 单尺度 : SSR
    - 多尺度 : MSR
    - 多尺度自适应增益 : MSRCR
-清晰度评价
    类ImgDefinition():清晰度评价算法
    类Fuzzy_image():模糊判断区间
    vagueJudge判断图像是否模糊,仅仅建议使用“L”或"G"算法
    - blog:https://blog.csdn.net/m0_62919535/article/details/127818006
"""
import cv2
from pyzjr.Showimage import StackedImages
from pyzjr.utils import empty
import random
import math
from pyzjr.FM import getPhotopath
import pyzjr.Z as Z
import numpy as np
import scipy.ndimage as ndi
from skimage.filters import sobel

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
                stackimg = StackedImages(0.5, [img, dst])
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

class Retinex():
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
        #将像素值归一化到0到255之间，并转换为无符号8位整型
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


class ImgDefinition():
    """
    Laplacian与Gradientmetric最靠谱
    """
    def Brenner(self, img):
        '''
        Brenner梯度函数
        :param img:narray 二维灰度图像
        :return: int 图像越清晰越大
        '''
        shapes = np.shape(img)
        output = 0
        for x in range(0, shapes[0] - 2):
            for y in range(0, shapes[1]):
                output += (int(img[x + 2, y]) - int(img[x, y])) ** 2
        return output

    def EOG(self,img):
        '''
        能量梯度函数(Energy of Gradient)
        :param img:narray 二维灰度图像
        :return: int 图像越清晰越大
        '''
        shapes = np.shape(img)
        output = 0
        for x in range(0, shapes[0] - 1):
            for y in range(0, shapes[1] - 1):
                output += ((int(img[x + 1, y]) - int(img[x, y])) ** 2 + (int(img[x, y + 1]) - int(img[x, y])) ** 2)
        return output

    def Roberts(self,img):
        '''
        Roberts函数
        :param img:narray 二维灰度图像
        :return: int 图像越清晰越大
        '''
        shapes = np.shape(img)
        output = 0
        for x in range(0, shapes[0] - 1):
            for y in range(0, shapes[1] - 1):
                output += ((int(img[x + 1, y + 1]) - int(img[x, y])) ** 2 + (
                        int(img[x + 1, y]) - int(img[x, y + 1])) ** 2)
        return output

    def Laplacian(self,img):
        '''
        Laplace
        :param img:narray 二维灰度图像
        :return: int 图像越清晰越大
        '''
        return cv2.Laplacian(img, Z.Lap_64F).var()

    def SMD(self,img):
        '''
        SMD(灰度方差)函数
        :param img:narray 二维灰度图像
        :return: int 图像越清晰越大
        '''
        shape = np.shape(img)
        output = 0
        for x in range(1, shape[0] - 1):
            for y in range(0, shape[1]):
                output += math.fabs(int(img[x, y]) - int(img[x, y - 1]))
                output += math.fabs(int(img[x, y] - int(img[x + 1, y])))
        return output

    def SMD2(self,img):
        '''
        （灰度方差乘积）函数
        :param img:narray 二维灰度图像
        :return: int 图像约清晰越大
        '''
        shape = np.shape(img)
        output = 0
        for x in range(0, shape[0] - 1):
            for y in range(0, shape[1] - 1):
                output += math.fabs(int(img[x, y]) - int(img[x + 1, y])) * math.fabs(
                    int(img[x, y] - int(img[x, y + 1])))
        return output

    def Gradientmetric(self,img, size=11):
        """
        计算指示图像中模糊强度的度量(0表示无模糊,1表示最大模糊)
        但实际上还是要根据我们的标准图与模糊图得到模糊区间
        :param img: 单通道进行处理,灰度图
        :param size: 重新模糊过滤器的大小
        :return: float,唯一不是值越大,越清晰的算法
        """
        image = np.array(img, dtype=np.float32) / 255
        n_axes = image.ndim
        shape = image.shape
        blur_metric = []

        slices = tuple([slice(2, s - 1) for s in shape])
        for ax in range(n_axes):
            filt_im = ndi.uniform_filter1d(image, size, axis=ax)
            im_sharp = np.abs(sobel(image, axis=ax))
            im_blur = np.abs(sobel(filt_im, axis=ax))
            T = np.maximum(0, im_sharp - im_blur)
            M1 = np.sum(im_sharp[slices])
            M2 = np.sum(T[slices])
            blur_metric.append(np.abs((M1 - M2)) / M1)

        return np.max(blur_metric) if len(blur_metric) > 0 else 0.0

    def get_method(self, mode):
        methods = {
            "B": self.Brenner,
            "E": self.EOG,
            "R": self.Roberts,
            "L": self.Laplacian,
            "S": self.SMD,
            "S2": self.SMD2,
            "G": self.Gradientmetric,
        }
        return methods.get(mode, None)

def definemode(img, mode):
    ti = ImgDefinition()
    method = ti.get_method(mode)

    if method is not None:
        return method(img)
    else:
        raise ValueError(f"[pyzjr]:Invalid mode: {mode}")

class Fuzzy_image():
    def __init__(self,mode="L"):
        """
        :param mode: 图像清晰度算法模式,建议采用“L”或“G”算法
        """
        self.mode=mode
    def getImgVar(self, image):
        """
        :param image: 图像
        :return: 图片清晰度数值
        """
        imggray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageVar = definemode(imggray,mode=self.mode)
        return abs(imageVar)

    def getReorder(self, imgfile, reverse = False):
        """
        :param imgfile: getPhotopath()获取图片路径的列表
        :param reverse: 表示是否进行反转排序(从大到小)
        :return: 模糊值排序
        """
        c = []
        for i in imgfile:
            img = cv2.imread(i)
            image = self.getImgVar(img)
            c.append(float(f"{image:.3f}"))
        c.sort(reverse=reverse)
        return c

    def getInterval(self, pathVague, pathStd):
        """
        :param pathVague:模糊图的数据集文件夹位置
        :param pathStd: 标准图的数据文件夹位置
        :return: 模糊判定区间
        """
        imgfile1, _ = getPhotopath(pathVague,debug=False)
        imgfile2, _ = getPhotopath(pathStd,debug=False)
        a = self.getReorder(imgfile1)
        b = self.getReorder(imgfile2)
        if self.mode=="G":
            thr = (b[-1], a[0])
        else:
            thr = (a[-1], b[0])
        return thr



def vagueJudge(img, pathVague, pathStd, mode="L", show_minandmax=True):
    """
    :param img: 图片
    :param show_minandmax:是否输出模糊未知区间
    :return: 模糊数值打印在图片上显示,只要没有超过清晰图像的阈值，全部判断为模糊
    """
    Fuzzy = Fuzzy_image(mode)
    imgVar = Fuzzy.getImgVar(img)
    minThr, maxThr = Fuzzy.getInterval(pathVague=pathVague, pathStd=pathStd)
    img2 = img.copy()
    if imgVar > maxThr:
        if mode == "G":
            text = "Vague"
            color = Z.red
        else:
            text = "Not Vague"
            color = Z.blue
    else:
        if mode == "G":
            text = "Not Vague"
            color = Z.blue
        else:
            text = "Vague"
            color = Z.red

    cv2.putText(img2, f"{text}{imgVar:.2f}", (12, 70), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

    if show_minandmax:
        print(minThr, maxThr)
    else:
        pass
    return img2

def main():
    path = r"ces\Standards\001.jpg"   # 需要进行测试的图片
    path2 = r"ces\test\02.jpg"
    pathVague = r"ces\test"           # 模糊图像数据
    pathStd = r"ces\Standards"        # 标准图像数据
    img = cv2.imread(path)
    test = vagueJudge(img, pathVague, pathStd, mode="G")
    cv2.imshow("test", test)
    cv2.waitKey(0)

def main2_Re():
    path= r"ces\Standards\001.jpg"
    img=cv2.imread(path)
    Re=Retinex()
    imgSSR=Re.SSR(img,5)
    imgMSR=Re.MSR(img, [1, 3, 5])
    imgMSRCR=Re.MSRCR(img,[1,3,5],12)
    imgStack=StackedImages(0.6,([img,imgSSR],[imgMSR,imgMSRCR]))
    cv2.imshow("retinex",imgStack)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
    main2_Re()


