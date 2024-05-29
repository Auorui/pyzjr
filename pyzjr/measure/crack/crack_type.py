import numpy as np
import cv2
from pyzjr.measure.pixel import SkeletonMap
from pyzjr.core.general import is_gray_image

# 裂缝类型
class CrackType():
    """直方图投影法推断裂缝类型, 能区分线性裂缝和网状裂缝"""
    def __init__(self, threshold=3, HWratio=10, Histratio=0.5, alpha=35, beta=55):
        """
        初始化分类裂缝的参数
        :param threshold: 阈值，用于分类裂缝的阈值
        :param HWratio: 高宽比，用于分类裂缝的高宽比阈值
        :param Histratio: 直方图比例，用于分类裂缝的直方图比例阈值
        :param alpha: 倾斜裂纹分类的较低阈值角度
        :param beta: 倾斜裂纹分类的上阈值角度
        """
        self.threshold = threshold
        self.HWratio = HWratio
        self.Histratio = Histratio
        self.types = {0: 'Horizontal',
                      1: 'Vertical',
                      2: 'Oblique',
                      3: 'Mesh'}
        self.alpha = alpha
        self.beta = beta

    def inference_minAreaRect(self, minAreaRect):
        """
        旋转矩形框长边与x轴的夹角.
        旋转角度 angle 是相对于图像水平方向的夹角，范围是 -90 到 +90 度.
        然而，一般情况下，我们习惯将角度定义为相对于 x 轴正方向的夹角，范围是 -180 到 +180 度.
        """
        w, h = minAreaRect[1]
        if w > h:
            angle = int(minAreaRect[2])
        else:
            angle = -(90 - int(minAreaRect[2]))
        return w, h, angle

    def classify(self, minAreaRect, skeleton_pts, HW):
        """
        针对当前crack instance，对其进行分类；
        主要利用了骨骼点双向投影直方图、旋转矩形框宽高比/角度；
        :param minAreaRect: 最小外接矩形框，[(cx, cy), (w, h), angle]；
        :param skeleton_pts: 骨骼点坐标；
        :param HW: 当前patch的高、宽；
        """
        H, W = HW
        w, h, angle = self.inference_minAreaRect(minAreaRect)
        if w / h < self.HWratio or h / w < self.HWratio:
            pts_y, pts_x = skeleton_pts[:, 0], skeleton_pts[:, 1]
            hist_x = np.histogram(pts_x, W)
            hist_y = np.histogram(pts_y, H)
            print("ss1",hist_x[0], hist_y[0])
            if self.hist_judge(hist_x[0]) and self.hist_judge(hist_y[0]):
                return 3

        return self.angle2cls(angle)

    def hist_judge(self, hist_v, esp=1e-5):
        less_than_T = np.count_nonzero((hist_v > 0) & (hist_v <= self.threshold))
        more_than_T = np.count_nonzero(hist_v > self.threshold)
        return more_than_T / (less_than_T + esp) > self.Histratio

    def angle2cls(self, angle):
        angle = abs(angle)
        assert 0 <= angle <= 90, "ERROR: The angle value exceeds the limit and should be between 0 and 90 degrees!"
        if angle < self.alpha:
            return 0
        elif self.alpha <= angle <= self.beta:
            return 2
        elif angle > self.beta:
            return 1
        else:
            return None

def _get_minAreaRect_information(mask):
    """
    从二值化掩膜图像中获取最小外接矩形的相关信息
    :param mask:二值化掩膜图像，包含目标区域的白色区域
    :return:最小外接矩形的信息，包括中心点坐标、宽高和旋转角度
    """
    if is_gray_image(mask):
        gray = mask
    else:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_merge = np.vstack(contours)
    minAreaRect = cv2.minAreaRect(contour_merge)
    return minAreaRect

def infertype(mask):
    """推导裂缝类型"""
    crack = CrackType()
    H, W = mask.shape[:2]
    minAreaRect = _get_minAreaRect_information(mask)
    skeimage, skepoints = SkeletonMap(mask)
    result = crack.classify(minAreaRect, skepoints, HW=(H, W))
    crack_type = crack.types[result]
    return crack_type

if __name__=="__main__":
    from pyzjr.measure.skeleton_extraction import (
        skeletoncv, sketion_medial_axis, sketionio)
    path = r'D:\PythonProject\pyzjrPyPi\models_img\1604.png'
    image_cv = skeletoncv(path)
    print(image_cv.shape)
    print(type(image_cv))

    image_io = sketionio(path)
    print(image_io.shape)
    print(type(image_io))

    image_axis = sketion_medial_axis(path)
    print(image_axis.shape)
    print(type(image_axis))

    crack_type = infertype(image_axis)
    print(crack_type)
