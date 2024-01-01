"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is used to implement clarity evaluation algorithms.
"""
import cv2
import math
import numpy as np
import scipy.ndimage as ndi
from skimage.filters import sobel

import pyzjr.Z as Z
from pyzjr.FM import getPhotopath

__all__ = ["ImgDefinition", "Fuzzy_image", "vagueJudge",]

class ImgDefinition():
    """
    清晰度评价函数: https://blog.csdn.net/m0_62919535/article/details/127818006
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
        模糊判定区间: https://blog.csdn.net/m0_62919535/article/details/128061017
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

if __name__=="__main__":
    path = r"ces\Standards\001.jpg"   # 需要进行测试的图片
    path2 = r"ces\test\02.jpg"
    pathVague = r"ces\test"           # 模糊图像数据
    pathStd = r"ces\Standards"        # 标准图像数据
    img = cv2.imread(path)
    test = vagueJudge(img, pathVague, pathStd, mode="G")
    cv2.imshow("test", test)
    cv2.waitKey(0)