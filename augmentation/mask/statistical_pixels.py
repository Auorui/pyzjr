"""
Copyright (c) 2023, Auorui.
All rights reserved.
"""
import cv2
import numpy as np
from pyzjr.augmentation.mask.contour import check_points_in_contour
from pyzjr.utils.check import is_gray_image

def unique(image):
    """Returns a list of unique pixel values in the input image"""
    np_image = np.array(image).astype(np.uint8)
    return np.unique(np_image)

def count_zero(thresh):
    """计算矩阵0值"""
    zero_count = np.sum(thresh == 0)
    return zero_count

def count_white(thresh):
    """计算矩阵255值"""
    white_count = np.sum(thresh == 255)
    return white_count

def count_nonzero(thresh):
    """计算矩阵非0值"""
    nonzero_pixels = np.count_nonzero(thresh)
    return nonzero_pixels

def incircleV1(image, contours_arr, is_draw=True, color=(0, 0, 255), use_check_point=False):
    """
    计算并绘制轮廓的最大内切圆。
    该方法通过对每个轮廓计算轮廓内部的最大内切圆，并返回最大内切圆的直径和位置。

    :param image: 单通道图像，通常是二值化后的图像，用于计算最大内切圆。
    :param contours_arr: 轮廓列表，通常通过 `cv2.findContours` 获得。每个轮廓是一个二维点集。
    :param is_draw: 是否在结果图像上绘制最大内切圆，默认为 True。
                     如果为 True，则在图像上绘制内切圆。
    :param color: 绘制内切圆的颜色，默认为红色 (0, 0, 255)。
    :param use_check_point: 是否进行内切圆验证，通过检查圆心及其半径内的点是否都在轮廓内部。
                             该参数主要适用于裂缝类型为网状结构时使用，默认值为 False。
    :return: 如果 `is_draw=True`，返回包含绘制内切圆的图像以及内切圆的直径。
             如果 `is_draw=False`，仅返回内切圆的直径。
    """
    if is_gray_image(image):
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result = image.copy()

    max_val = 0
    max_dist_pt = None
    # 只考虑非零点
    non_zero_points = np.array(np.nonzero(image))
    non_zero_points = non_zero_points.T
    for contours in contours_arr:
        raw_dist = np.zeros(image.shape[:2], dtype=np.float32)
        # 使用cv2.pointPolygonTest计算每个非零点到轮廓的距离
        for point in non_zero_points:
            x, y = point
            x, y = int(x), int(y)
            raw_dist[x, y] = cv2.pointPolygonTest(contours, (y, x), True)
        # for i in range(image.shape[0]):
        #     for j in range(image.shape[1]):
        #         raw_dist[i, j] = cv2.pointPolygonTest(contours, (j, i), True)
        # 寻找最大内切圆直径及其圆心
        min_val, curr_max_val, _, curr_max_dist_pt = cv2.minMaxLoc(raw_dist)
        # 确保选中的点在轮廓内部
        if curr_max_val > max_val:
            # 检查圆心加上半径后的八个方向点是否都在轮廓内部
            if use_check_point:
                if check_points_in_contour(curr_max_dist_pt, int(curr_max_val)-2, contours):
                    max_val = curr_max_val
                    max_dist_pt = curr_max_dist_pt
            else:
                max_val = curr_max_val
                max_dist_pt = curr_max_dist_pt
    wide = max_val * 2
    if is_draw and max_dist_pt is not None:
        result = cv2.circle(result, max_dist_pt, int(max_val), color, 1, 1, 0)

    if is_draw:
        return wide, result
    else:
        return wide

def incircleV2(image, is_draw=True, color=(0, 0, 255)):
    """
    计算并绘制图像的最大内切圆。
    该方法通过距离变换 (`cv2.distanceTransform`) 来计算图像的最大内切圆的直径和位置。

    :param image: 单通道图像，通常是二值化后的图像，用于计算最大内切圆。
    :param is_draw: 是否在结果图像上绘制最大内切圆，默认为 True。
                     如果为 True，则在图像上绘制内切圆。
    :param color: 绘制内切圆的颜色，默认为红色 (0, 0, 255)。
    :return: 如果 `is_draw=True`，返回包含绘制内切圆的图像以及内切圆的直径。
             如果 `is_draw=False`，仅返回内切圆的直径。
    """
    if is_gray_image(image):
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result = image.copy()

    max_val = 0
    max_dist_pt = None
    dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 3)
    min_val, curr_max_val, _, curr_max_dist_pt = cv2.minMaxLoc(dist_transform)
    if curr_max_val > max_val:
        max_val = curr_max_val
        max_dist_pt = curr_max_dist_pt
    wide = max_val * 2
    if is_draw and max_dist_pt is not None:
        result = cv2.circle(result, max_dist_pt, int(max_val), color, 1, 1, 0)
    if is_draw:
        return wide, result
    else:
        return wide

def outcircle(img, contours_arr, is_draw=True, color=(0, 255, 0)):
    """
    轮廓外切圆算法,画出所有轮廓的外切圆
    :param img: 单通道图像
    :param contours_arr: ndarry的轮廓, 建议使用cv2.findContours,
                        contours_arr, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        pyzjr也有SearchOutline可用,但要转换成ndarry的格式
    :example:
        contour = [np.array([point], dtype=np.int32) for point in contours]
    :param color: 绘制外切圆颜色
    :return: 包含结果图像（如果is_draw为True）和外切圆信息（所有轮廓的外切圆圆心和半径）的元组。
           如果is_draw为False，则只返回外切圆信息。
    """
    radii = []
    if is_draw:
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        result = None

    for cnt in contours_arr:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        radii.append([(x, y), radius])
        if is_draw:
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(result, center, radius, color, 1)  # 绘制外切圆
            cv2.circle(result, center, 1, color, 1)       # 绘制圆心

    if is_draw:
        return radii, result
    else:
        return radii


if __name__ == "__main__":
    from pyzjr.augmentation.mask.predeal import binarization
    image = cv2.imread(r"E:\PythonProject\pyzjrPyPi\pyzjr\measure\fissure\0003.png")
    thresh = binarization(image)
    # contours_arr = SearchOutline(thresh)
    contours_arr, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    wide, result = incircleV1(thresh, contours_arr)
    print(wide)
    wide, result_transform = incircleV2(thresh)
    print(wide)
    # radii, results = outcircle(thresh, contours_arr)
    # print(wide, radii)
    # cv2.imwrite("sss.png", result_transform)
    cv2.imshow("ss", result_transform)
    cv2.waitKey(0)
