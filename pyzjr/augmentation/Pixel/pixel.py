import numpy as np
import cv2
import string
from skimage.morphology import skeletonize, disk
from skimage.filters import threshold_otsu, rank
from skimage.color import rgb2gray
from pyzjr.core import is_gray_image

__all__=["SkeletonMap","incircle","outcircle"]

def SkeletonMap(target):
    """
    获取骨架图的信息
    :param target: 目标图
    :return: 骨架图与一个数组，其中每一行表示一个非零元素的索引(y,x)，包括行索引和列索引
    """
    gray = rgb2gray(target)
    thresh = threshold_otsu(gray)
    binary = gray > thresh
    skimage = skeletonize(binary)
    skepoints = np.argwhere(skimage)
    skimage = skimage.astype(np.uint8)
    return skimage, skepoints


def incircle(img, contours_arr, color=(0, 0, 255)):
    """
    轮廓最大内切圆算法,所有轮廓当中的内切圆
    :param img: 单通道图像
    :param contours_arr: ndarry的轮廓, 建议使用cv2.findContours,
                        contours_arr, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        pyzjr也有SearchOutline可用,但要转换成ndarry的格式
    :example:
        contour = [np.array([point], dtype=np.int32) for point in contours]
        # 平铺的方法
        flatten_contours = np.concatenate([cnt.flatten() for cnt in contours_arr])
        flatten_contours = flatten_contours.reshape(-1, 2)
    :param color: 绘制内切圆颜色
    :return: 绘制在原图的内切圆,内切圆直径,绘制出的图像与轮廓直接差距一个像素，是因为cv2.circle的半径参数必须为int类型
    """
    if is_gray_image:
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    raw_dist = np.zeros(img.shape, dtype=np.float32)
    letters = list(string.ascii_uppercase)
    label = {}
    for k, contours in enumerate(contours_arr):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                raw_dist[i, j] = cv2.pointPolygonTest(contours, (j, i), True)
        min_val, max_val, _, max_dist_pt = cv2.minMaxLoc(raw_dist)
        label[letters[k]] = max_val * 2
        radius = int(max_val)
        cv2.circle(result, max_dist_pt, radius, color, 1, 1, 0)

    return result, label


def outcircle(img, contours_arr, color=(0,255,0)):
    """
    轮廓外切圆算法,画出所有轮廓的外切圆
    :param img: 单通道图像
    :param contours_arr: ndarry的轮廓, 建议使用cv2.findContours,
                        contours_arr, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        pyzjr也有SearchOutline可用,但要转换成ndarry的格式
    :example:
        contour = [np.array([point], dtype=np.int32) for point in contours]
    :param color: 绘制外切圆颜色
    :return:
    """
    radii = []
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for k, cnt in enumerate(contours_arr):
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        radii.append([(x,y),radius])
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(result, center, radius, color, 1)
        cv2.circle(result, center, 1, color, 1)

    return result, radii


class ToMask():
    """集成不同处理掩膜、标签的方式"""
    def __init__(self, mask):
        self.mask = mask

    def Binary(self, mask):
        pass

    def up_low(self, lower, upper):
        """
        :param lower: 颜色范围的下限值，作为一个包含(B, G, R)值的元组
        :param upper: 颜色范围的上限值，作为一个包含(B, G, R)值的元组
        """
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(self.mask, lower, upper)

        return mask

    def threshold(self, std=127.5):
        """阈值法"""
        self.mask[self.mask > std] = 255
        self.mask[self.mask < std] = 0
        mask = self.mask.astype("uint8")

        return mask

    def combine(self, operation='and', *masks):
        """
        :param operation: 组合操作，可选 'and', 'or', 'not'
        :param masks: 要组合的掩膜列表
        """
        if not masks:
            return

        combined_mask = self.mask.copy()

        for mask in masks:
            if operation == 'and':
                combined_mask = cv2.bitwise_and(combined_mask, mask)
            elif operation == 'or':
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            elif operation == 'not':
                combined_mask = cv2.bitwise_not(mask)
            else:
                raise ValueError("Invalid operation. Use 'and', 'or', or 'not'.")

        return combined_mask



