import cv2
import numpy as np
from pyzjr.ColorModule import ColorFinder
import pyzjr.utils as zjr
import logging

def HSV(mode):
    """
    * 示例仅供参考，只是为了在调试过程中能够更快的找到相要的颜色
     可以进行调试成功的值进行替换，但也要记住将原来颜色的值注释。
    :param mode:
    :return:
    """
    if mode == 'black':
        Min_and_MaxVal = [[0, 0, 0], [180, 255, 46]]
        return Min_and_MaxVal
    elif mode == 'gray':
        Min_and_MaxVal = [[0, 0, 46], [180, 43, 220]]
        return Min_and_MaxVal
    elif mode == 'white':
        Min_and_MaxVal = [[0, 0, 221], [180, 30, 225]]
        return Min_and_MaxVal
    elif mode == 'red':
        Min_and_MaxVal = [[156, 43, 46], [180, 255, 225]]
        # Min_and_MaxVal = [156, 43, 46], [180, 255, 225] or [0, 43, 46], [10, 255, 225]
        return Min_and_MaxVal
    elif mode == 'orange':
        Min_and_MaxVal = [[11, 43, 46], [25, 255, 225]]
        return Min_and_MaxVal
    elif mode == 'yellow':
        Min_and_MaxVal = [[26, 43, 46], [34, 255, 225]]
        return Min_and_MaxVal
    elif mode == 'green':
        Min_and_MaxVal = [[28,38,36],[64,255,255]]
        # 参考值：[[35, 43, 46], [77, 255, 225]]
        return Min_and_MaxVal
    elif mode == 'cyan':
        Min_and_MaxVal = [[78, 43, 46], [99, 255, 225]]
        return Min_and_MaxVal
    elif mode == 'blue':
        Min_and_MaxVal = [[100, 43, 46], [124, 255, 225]]
        return Min_and_MaxVal
    elif mode == 'purple':
        Min_and_MaxVal = [[125, 43, 46], [155, 255, 225]]
        return Min_and_MaxVal
    elif mode == 0:
        print("请按照‘,’分隔分别输入hmin, smin, vmin, hmax, smax, vmax的值", end='/n')
        return 0
    else:
        logging.warning("颜色并不在示例范围内")




class getMask():
    def __init__(self, trackBar=True):
        self.ColF = ColorFinder(trackBar)
        # self.ColF.initTrackbars()
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
        :param HsvVals: 可以通过getTrackbarValues获得
        :return: 返回融合图、掩膜图、HSV图
        """
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array(HsvVals[0])
        upper = np.array(HsvVals[1])
        mask = cv2.inRange(imgHSV, lower, upper)
        imgResult = cv2.bitwise_and(img, img, mask=mask)
        return imgResult, mask, imgHSV

    def DetectImg(self, path, ConsoleOut=True, threshold=None, scale=1.0):
        """
        * 轨迹栏检测图片
        :param path: 图片路径
        :param ConsoleOut: 用于是否控制台打印HsvVals的值
        :param threshold: 阈值，用于保护图片的区域
        :param scale: 图片规模大小
        :return: 无返回
        """
        while True:
            img = cv2.imread(path)
            HsvVals = self.ColF.getTrackbarValues(False)
            if ConsoleOut:
                print(HsvVals)
            imgResult, mask, imgHSV = self.MaskZone(img,HsvVals)
            pro_mask = self.protect_region(mask, threshold)
            imgStack = zjr.stackImages(scale, ([img,imgHSV],[pro_mask,imgResult]))
            cv2.imshow("Stacked Images", imgStack)
            k = cv2.waitKey(1)
            if k == 27:
                break


if __name__=="__main__":
    path = r'E:\pythonProject2\pyzjr\resources\AI.png'
    img2 = cv2.imread(path)
    getMask=getMask()
    getMask.DetectImg(path,threshold=[0, img2.shape[1], 300, img2.shape[0]],scale=0.4)





