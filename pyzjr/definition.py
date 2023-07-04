"""
- author:Auorui(夏天是冰红茶)
- creation time:2022.11.19
- blog:https://blog.csdn.net/m0_62919535/article/details/127818006
"""
import cv2
import math
from pyzjr.PIC import getPhotopath
import numpy as np
import pyzjr.Z as Z

class Clear_quantification():
    def brenner(self,img):
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
        SMD（灰度方差）函数
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

class Fuzzy_image():
    def getImgVar(self,image):
        """
        :param image: 图像
        :return: 图片清晰度数值
        """
        quanti=Clear_quantification()
        imggray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageVar = quanti.Laplacian(imggray)
        return imageVar

    def getTest(self,imgfile):
        """
        :param imgfile: getPhotopath（）
        :return: 模糊值排序
        """
        c = []
        for i in imgfile:
            # print(i)
            img = cv2.imread(i)
            image = self.getImgVar(img)
            # print(image)
            c.append(float(f"{image:.3f}"))
        if 'test' in imgfile[0]:  # 对测试集数据进行反转
            c.sort(reverse=True)
        else:
            c.sort()
        return c

    def getThr(self,pathTest = "./ces/test",pathStd = "./ces/Standards"):
        """
        :param pathTest:测试的数据集文件夹位置
        :param pathStd: 标准图的数据文件夹位置
        :return: 模糊判定未知区间
        """
        imgfile1 = getPhotopath(pathTest)
        imgfile2 = getPhotopath(pathStd)
        a = self.getTest(imgfile1)
        b = self.getTest(imgfile2)
        thr = (a[0], b[0])
        # print(thr)
        return thr

def vagueJudge(image, show_minandmax=True):
    """
    :param image: 图片
    :param show_minandmax:是否输出模糊未知区间
    :return: 模糊数值打印在图片上显示
    """
    Fuzzy = Fuzzy_image()
    img = cv2.imread(image)
    imgVar = Fuzzy.getImgVar(img)
    minThr, maxThr = Fuzzy.getThr()
    if imgVar > maxThr:
        cv2.putText(img, f"Not Vague{imgVar:.2f}", (12, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 3)
    else:
        cv2.putText(img, f"Vague{imgVar:.2f}", (12, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 3)
    if show_minandmax:
        print(minThr, maxThr)
    else:
        pass
    cv2.imshow("img", img)
    k = cv2.waitKey(0) & 0xFF


if __name__ == "__main__":
    image="./test/01.jpg"   #需要进行测试的图片
    vagueJudge(image)
