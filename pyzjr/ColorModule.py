import cv2
import numpy as np
import logging
from pyzjr.utils import empty
from pyzjr.Showimage import StackedImages
from pyzjr.video import VideoCap
import pyzjr.Z as Z

class ColorFind():
    def __init__(self, trackBar=False, name="Bars"):
        self.trackBar = trackBar
        self.name = name
        if self.trackBar:
            self.initTrackbars()

    def initTrackbars(self):
        """
        :return:初始化轨迹栏
        """
        cv2.namedWindow(self.name)
        cv2.resizeWindow(self.name, 640, 240)
        cv2.createTrackbar("Hue Min", self.name, 0, 179, empty)
        cv2.createTrackbar("Hue Max", self.name, 179, 179, empty)
        cv2.createTrackbar("Sat Min", self.name, 0, 255, empty)
        cv2.createTrackbar("Sat Max", self.name, 255, 255, empty)
        cv2.createTrackbar("Val Min", self.name, 0, 255, empty)
        cv2.createTrackbar("Val Max", self.name, 255, 255, empty)

    def getTrackbarValues(self):
        """
        Gets the trackbar values in runtime
        :return: hsv values from the trackbar window
        """
        hmin = cv2.getTrackbarPos("Hue Min", self.name)
        smin = cv2.getTrackbarPos("Sat Min", self.name)
        vmin = cv2.getTrackbarPos("Val Min", self.name)
        hmax = cv2.getTrackbarPos("Hue Max", self.name)
        smax = cv2.getTrackbarPos("Sat Max", self.name)
        vmax = cv2.getTrackbarPos("Val Max", self.name)
        HsvVals=[[hmin, smin, vmin],[hmax, smax, vmax]]

        return HsvVals

    def protect_region(self, mask ,threshold=None):
        """
        * 用于保护掩膜图的部分区域
        :param mask: 掩膜图
        :param threshold: 如果为None,则为不保护，如果是长为4的列表，则进行特定区域的保护
        :return: 返回进行保护区域的掩膜图

        example:    [0, img.shape[1], 0, img.shape[0]]为全保护状态，
                    x_start可以保护大于x的部分
                    x_end可以保护小于x的部分
                    y_start可以保护图像下方的部分
                    y_end可以保护图像上方的部分
        """
        if threshold == None:
            return mask
        else:
            x_start, x_end, y_start, y_end = threshold[:4]
            mask[y_start:y_end, x_start:x_end] = 0
            return mask

    def MaskZone(self, img, HsvVals):
        """
        * 生成掩膜图以及融合图像
        :param img: 输入图像
        :param HsvVals: 可以通过getTrackbarValues获得,也可调取Z.HSV的值
        :return: 返回融合图、掩膜图、HSV图
        """
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array(HsvVals[0])
        upper = np.array(HsvVals[1])
        mask = cv2.inRange(imgHSV, lower, upper)
        imgResult = cv2.bitwise_and(img, img, mask=mask)
        return imgResult, mask

    def update(self, img, myColor=None):
        """
        :param img: 需要在其中找到颜色的图像
        :param myColor: hsv上下限列表
        :return: mask带有检测到颜色的白色区域的roi图像
                 imgColor彩色图像仅显示检测到的区域
        """
        imgColor = [],
        mask = []
        if self.trackBar:
            myColor = self.getTrackbarValues()

        if isinstance(myColor, str):
            myColor = self.getColorHSV(myColor)

        if myColor is not None:
            imgColor, mask = self.MaskZone(img,myColor)
        return imgColor, mask

    def getColorHSV(self, myColor):
        if myColor == 'red':
            output = [[146, 141, 77], [179, 255, 255]]
        elif myColor == 'green':
            output = [[44, 79, 111], [79, 255, 255]]
        elif myColor == 'blue':
            output = [[103, 68, 130], [128, 255, 255]]
        else:
            output = None
            logging.warning("Color Not Defined")
            logging.warning("Available colors: red, green, blue ")

        return output

def DetectImg(img, name="DetectImg", ConsoleOut=True, threshold=None, scale=1.0):
    """
    * 轨迹栏检测图片,此函数仅仅作为使用示例
    :param img: 图片
    :param name: 轨迹栏名
    :param ConsoleOut: 用于是否控制台打印HsvVals的值
    :param threshold: 阈值，用于保护图片的区域
    :param scale: 规模大小
    :return:
    """
    ColF = ColorFind(True, name)
    while True:
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        HsvVals = ColF.getTrackbarValues()
        if ConsoleOut:
            print(HsvVals)
        imgResult, mask = ColF.update(img,HsvVals)
        pro_mask = ColF.protect_region(mask, threshold)
        imgStack = StackedImages(scale, ([img,imgHSV],[pro_mask,imgResult]))
        cv2.imshow("Stacked Images", imgStack)
        k = cv2.waitKey(1)
        if k == Z.Esc:
            break

def DetectVideo(name="DetectVideo",mode=0,myColor=None,scale=1.0):
    """
    * 轨迹栏检测摄像头,此函数仅仅作为使用示例
    :param mode: 检测模式,默认本地摄像头,可传入video路径
    :param name: 轨迹栏名
    :param myColor: getColorHSV返回的一些测好的Hsv值
    :param scale: 规模大小
    :return:
    """
    if myColor:
        Cf = False
    else:
        Cf = True
    Vcap = VideoCap()
    Vcap.CapInit(mode=mode)
    ColF = ColorFind(Cf, name)
    while True:
        img = Vcap.read()
        imgColor, mask = ColF.update(img, myColor)
        stackedimg=StackedImages(scale,[img,imgColor])
        cv2.imshow(name, stackedimg)
        if cv2.waitKey(1) & 0xFF == Z.Esc:
            break

if __name__=="__main__":
    path = r'ces\test\03.jpg'
    img = cv2.imread(path)
    # DetectImg(img,scale=0.4)
    DetectVideo(myColor="red")

