import numpy as np
import cv2
import string
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from skimage import morphology, measure

from pyzjr.utils.check import is_gray_image
from pyzjr.augmentation.contour import SearchOutline
from pyzjr.visualize.cvplot import PutBoxText

def SkeletonMap(target):
    """
    获取骨架图的信息
    :param target: 目标图
    :return: 骨架图与一个数组，其中每一行表示一个非零元素的索引(y,x)，包括行索引和列索引
    """
    if target.ndim == 2 or target.shape[2] == 1:
        gray = target
    else:
        gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    thresh = threshold_otsu(target)
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
    k = 0
    for contours in contours_arr:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                raw_dist[i, j] = cv2.pointPolygonTest(contours, (j, i), True)
        min_val, max_val, _, max_dist_pt = cv2.minMaxLoc(raw_dist)
        if max_val > .5:
            label[letters[k]] = max_val * 2
            k += 1
        cv2.circle(result, max_dist_pt, int(max_val), color, 1, 1, 0)

    return result, label


def outcircle(img, contours_arr, color=(0, 255, 0)):
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


def get_each_crack_areas(mask, thresh, merge_threshold=3, area_threshold=50):
    """每条裂缝的面积,并用大写字母来进行标记"""
    connected_image = morphology.closing(thresh, morphology.disk(merge_threshold))
    labeled_image = measure.label(connected_image, connectivity=2)
    region_props = measure.regionprops(labeled_image)
    area = {}
    Bboxing = []
    crack_label = ord('A')
    for region in region_props:
        area_value = region.area
        if area_value >= area_threshold:
            minr, minc, maxr, maxc = region.bbox
            Bboxing.append([(minc, minr), (maxc, maxr)])
            PutBoxText(mask, [minc, minr, maxc, maxr], chr(crack_label), mode=0, fontsize=.5)
            if crack_label <= ord('Z'):
                area[chr(crack_label)] = area_value
                crack_label += 1
    return area, Bboxing, mask

if __name__=="__main__":
    from pyzjr.measure.skeleton_extraction import skeletoncv
    path = r'D:\PythonProject\pyzjrPyPi\models_img\1604.png'
    cv_image = cv2.imread(path)
    image = skeletoncv(path)
    # image = BinaryImg(cv_image)
    contour_lengths, all_length = foreground_contour_length(image)

    print(all_length)
    for i, length in enumerate(contour_lengths):
        print(f"Contour {i} length: {length}")