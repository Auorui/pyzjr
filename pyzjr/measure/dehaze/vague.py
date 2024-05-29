import cv2

import pyzjr.Z as Z
from pyzjr.data.file import getPhotopath
from pyzjr.measure.dehaze.eval_definition import Laplacian, Gradientmetric

class Fuzzy_image():
    def __init__(self, mode="Laplacian"):
        """
        模糊判定区间: https://blog.csdn.net/m0_62919535/article/details/128061017
        :param mode: 图像清晰度算法模式,建议采用“Laplacian”或“Gradientmetric”算法,
                    其他的效果差
        """
        self.mode = mode
        if mode=='Laplacian':
            self.definemode = Laplacian
        elif mode=='Gradientmetric':
            self.definemode = Gradientmetric

    def getImgVar(self, image):
        """
        :param image: 图像
        :return: 图片清晰度数值
        """
        imggray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageVar = self.definemode(imggray)
        return abs(imageVar)

    def getReorder(self, imgfile, reverse=False):
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
        if self.mode == "Gradientmetric":
            thr = (b[-1], a[0])
        else:
            thr = (a[-1], b[0])
        return thr

def vagueJudge(img, pathVague, pathStd, mode="Laplacian", show_minandmax=True):
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
    return img2

if __name__=="__main__":
    path = r"ces\Standards\001.jpg"   # 需要进行测试的图片
    path2 = r"ces\test\02.jpg"
    pathVague = r"ces\test"           # 模糊图像数据
    pathStd = r"ces\Standards"        # 标准图像数据
    img = cv2.imread(path)
    test = vagueJudge(img, pathVague, pathStd, mode="Laplacian")
    cv2.imshow("test", test)
    cv2.waitKey(0)