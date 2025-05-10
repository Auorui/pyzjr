"""
Copyright (c) 2023, Auorui.
All rights reserved.
"""
import cv2
from pylab import *
from skimage.morphology import disk
from skimage.filters import rank
from skimage import measure
from pyzjr.augmentation.mask.predeal import binarization
import pyzjr.Z as Z

def getContours(img, cThr=(50, 100), minArea=1000, filter=0, draw=True):
    """
    :param img: 输入图像
    :param cThr: 阈值
    :param minArea: 更改大小
    :param filter: 过滤
    :param draw: 绘制边缘
    :return: 返回图像轮廓
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv2.erode(imgDial, kernel, iterations=2)
    contours, _ = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append([len(approx), area, approx, bbox, i])
            else:
                finalCountours.append([len(approx), area, approx, bbox, i])
    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalCountours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)
    return img, finalCountours

def SearchOutline(binary_image):
    """
    在给定图像中搜索轮廓并返回轮廓的坐标列表。
    :param binary_image: 要搜索轮廓的图像。
    :return: 包含轮廓坐标的列表，每个坐标表示裂缝的一个点，坐标格式为 [(x, y),...]。
    """
    contours = measure.find_contours(binary_image, level=128, fully_connected='low', positive_orientation='low')
    contours_xy = [np.fliplr(np.vstack(contour)).astype(np.int32) for contour in contours]
    return contours_xy

def drawOutline(blackbackground, contours, color=Z.purple, thickness=1):
    """绘制轮廓"""
    cv2.drawContours(blackbackground, contours, -1, color, thickness=thickness)

def gradientOutline(img, radius=2):
    """
    对图像进行梯度边缘检测，并返回边缘强度图像。
    
    :param img: 输入的图像(内部有二值转化)
    :param radius: 半径,默认为2。
    :return: 返回经过梯度边缘检测处理后的边缘强度图像
    """
    image = binarization(img)
    denoised = rank.median(image, disk(radius))
    gradient = rank.gradient(denoised, disk(radius))
    gradient = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

    return gradient

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes


def label_contour(image, c, i=1, prefix="#", color=(0, 255, 0), thickness=2):
    """
    在图像上标记轮廓，并在轮廓中心显示标签。

    参数:
        image: 输入的图像（numpy.ndarray）。
        c: 轮廓点集（numpy.ndarray）。
        i: 轮廓的索引值（int）。
        prefix: 标签的前缀，默认为 "#"（str）。
        color: 轮廓的颜色，默认为绿色 (0, 255, 0)（tuple）。
        thickness: 轮廓线条的粗细，默认为 2（int）。
    返回:
        标记了轮廓和标签的图像。
    """
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    cv2.drawContours(image, [c], -1, color, thickness)
    cv2.putText(image, f"{prefix}{i}", (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)
    return image


def foreground_contour_length(binary_img, minArea=5):
    """
    计算前景的轮廓长度, 返回两个值, 每个轮廓的长度和总长度
    :param binary_img: 二值图
    :param minArea: 最小过滤面积
    """
    contours_xy = SearchOutline(binary_img)
    contour_lengths = []
    for contour in contours_xy:
        area = cv2.contourArea(contour)
        if area > minArea:
            length = cv2.arcLength(contour, False)
            contour_lengths.append(length)
    all_length = np.sum(contour_lengths)
    return contour_lengths, all_length


def check_points_in_contour(center, radius, contour):
    x, y = center
    directions = [
        (x + radius, y),  # 右
        (x - radius, y),  # 左
        (x, y + radius),  # 下
        (x, y - radius),  # 上
        (x + radius, y + radius),  # 右下
        (x + radius, y - radius),  # 右上
        (x - radius, y + radius),  # 左下
        (x - radius, y - radius),  # 左上
    ]
    for pt in directions:
        if cv2.pointPolygonTest(contour, pt, False) < 0:  # 点在轮廓外
            return False
    return True


if __name__=="__main__":
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    contour = np.array([[100, 100], [400, 100], [400, 400], [100, 400]], dtype=np.int32)
    labeled_image = label_contour(image, contour, 0)
    cv2.imshow("Labeled Contour", labeled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()