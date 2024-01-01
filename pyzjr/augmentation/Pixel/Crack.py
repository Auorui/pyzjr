"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is used to quantify two-dimensional information such as the number, length, width, and area of cracks
in the semantic segmentation graph of neural networks.
"""
import cv2
import numpy as np

import string
import os
from skimage.filters import threshold_otsu,median
from skimage.morphology import skeletonize,dilation,disk
from skimage import io, morphology, measure

from pyzjr.augmentation.Pixel import SkeletonMap, incircle
from pyzjr.augmentation.Color import putBoxText


# 裂缝类型
class CrackType():
    """直方图投影法推断裂缝类型,仅仅能区分线性裂缝和网状裂缝"""
    def __init__(self, threshold=3, HWratio=10, Histratio=0.5):
        """
        初始化分类裂缝的参数
        :param threshold: 阈值，用于分类裂缝的阈值
        :param HWratio: 高宽比，用于分类裂缝的高宽比阈值
        :param Histratio: 直方图比例，用于分类裂缝的直方图比例阈值
        """
        self.threshold = threshold
        self.HWratio = HWratio
        self.Histratio = Histratio
        self.types = {0: 'Horizontal',
                      1: 'Vertical',
                      2: 'Oblique',
                      3: 'Mesh'}

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
            if self.hist_judge(hist_x[0]) and self.hist_judge(hist_y[0]):
                return 3

        return self.angle2cls(angle)

    def hist_judge(self, hist_v):
        less_than_T = np.count_nonzero((hist_v > 0) & (hist_v <= self.threshold))
        more_than_T = np.count_nonzero(hist_v > self.threshold)
        return more_than_T / (less_than_T + 1e-5) > self.Histratio

    @staticmethod
    def angle2cls(angle):
        angle = abs(angle)
        assert 0 <= angle <= 90, "ERROR: The angle value exceeds the limit and should be between 0 and 90 degrees!"
        if angle < 35:
            return 0
        elif 35 <= angle <= 55:
            return 2
        elif angle > 55:
            return 1
        else:
            return None

def _get_minAreaRect_information(mask):
    """
    从二值化掩膜图像中获取最小外接矩形的相关信息
    :param mask:二值化掩膜图像，包含目标区域的白色区域
    :return:最小外接矩形的信息，包括中心点坐标、宽高和旋转角度
    """
    gray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
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

class Crack_areas_and_numbers():
    def __init__(self, mask, t=2):
        self.imgMask = mask.copy()
        gray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, self.thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        self.Bboxing=[]
        self.t = t

    def Totalareas(self):
        """目标像素白色的个数"""
        image_array = np.array(self.thresh)
        white_pixels = np.count_nonzero(image_array)
        return white_pixels

    def Eachareas(self, merge_threshold=3, area_threshold=50):
        """每条裂缝的面积,并用大写字母来进行标记"""
        connected_image = morphology.closing(self.thresh, morphology.disk(merge_threshold))
        labeled_image = measure.label(connected_image, connectivity=2)
        region_props = measure.regionprops(labeled_image)
        self.area = {}
        crack_label = ord('A')
        for region in region_props:
            area_value = region.area
            if area_value >= area_threshold:
                minr, minc, maxr, maxc = region.bbox
                self.Bboxing.append([(minc, minr), (maxc, maxr)])
                putBoxText(self.imgMask, [minc, minr, maxc, maxr],chr(crack_label))
                if crack_label <= ord('Z'):
                    self.area[chr(crack_label)] = area_value
                    crack_label += 1
        return self.area, self.Bboxing, self.imgMask

    def Cracknums(self):
        """裂缝个数"""
        return len(self.area)

def Pixel_Coords(img, val=255):
    """用于计算所有目标像素的坐标位置,默认为白色"""
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    y_coords, x_coords = np.where(thresh == val)
    coords_list = list(zip(x_coords, y_coords))
    return coords_list

def reorder(coord, type):
    """np.where默认是按照从上到下进行查找的"""
    if type == "Horizontal":
        coords = sorted(coord, key=lambda x: x[0])
        return coords
    elif type == "Vertical" or type == "Oblique":
        return coord

def Crop_cracks_img(mask, Bboxing):
    """根据边界框,进行裁剪,并存入列表中"""
    cropped_cracks = []
    for bbox in Bboxing:
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]
        cropped_crack = mask[y1:y2, x1:x2]
        cropped_cracks.append(cropped_crack)
    return cropped_cracks

def Crack_label(cracknums):
    """创建一个关于A-Z的标签"""
    labels = []
    for i in range(cracknums):
        label = string.ascii_uppercase[i]
        labels.append(label)
    return labels

def Nan():
    """仅表示一个无效值"""
    return 'Nan'

def Classify_points_basedon_Bbox(gujia_pos, Bboxing):
    """按照边界框对像素进行分类"""
    classified_gujia_pos = {}
    for point in gujia_pos:
        x, y = point
        for i, bbox in enumerate(Bboxing):
            (minc, minr), (maxc, maxr) = bbox
            if minc <= x <= maxc and minr <= y <= maxr:
                category_label = chr(ord('A') + i)
                if category_label not in classified_gujia_pos:
                    classified_gujia_pos[category_label] = []
                classified_gujia_pos[category_label].append(point)
                break
    return classified_gujia_pos

def Crack_of_length(coords_lst, bias = 1.118):
    """这里用于计算裂缝的长度"""
    connection, inflexion, spaceBetween = 0, 0, 0
    for i in range(len(coords_lst) - 1):
        start = coords_lst[i]
        end = coords_lst[i + 1]
        gap = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
        if gap == 1.0:
            connection += 1
        elif gap == np.sqrt(2):
            inflexion += 1
        spaceBetween += gap
    return spaceBetween + bias, inflexion, connection

def Crack_of_width(total_area, total_length):
    """求解裂缝平均宽度"""
    ave_width = total_area / (total_length + 1e-5)
    return ave_width

class Crack_skeleton():
    def sketioncv(self, single_pic):
        image = cv2.imread(single_pic, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=1)
        binary = cv2.medianBlur(binary, 5)
        binary = cv2.dilate(binary, kernel, iterations=1)
        binary = cv2.medianBlur(binary, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary=cv2.cvtColor(binary,cv2.COLOR_BGR2RGB)
        skeleton = skeletonize(binary)
        skeleton=cv2.cvtColor(skeleton,cv2.COLOR_RGB2BGR)
        return skeleton

    def sketionio(self,single_pic):
        image = io.imread(single_pic, as_gray=True)
        thresh = threshold_otsu(image)
        binary = image > thresh

        binary = dilation(binary, disk(3))
        binary = median(binary, footprint=morphology.disk(5))
        binary = dilation(binary, disk(2))
        binary = median(binary, footprint=morphology.disk(5))

        selem = morphology.disk(3)
        binary = morphology.closing(binary, selem)

        skeleton = skeletonize(binary)
        return skeleton

    def sketions(self, input_folder='num', output_folder='output'):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)  # 如果输出文件夹不存在，就创建它
        for filename in os.listdir(input_folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                input_file=os.path.join(input_folder, filename)
                output_filename = os.path.join(output_folder, filename)
                skeleton=self.sketionio(input_file)
                io.imsave(output_filename, skeleton)
        return output_folder

def DetectCracks(mask, skeimg, mode, merge_threshold=3, area_threshold=50, showcrackinformation=True):
    crack_area = Crack_areas_and_numbers(mask)
    cracktype = infertype(mask)
    if cracktype == "Mesh":
        total_areas = crack_area.Totalareas()
        total_length = width = CrackNum = crack_information = Nan()
        return CrackNum, total_areas, total_length, width, crack_information

    else:
        area, Bboxing, Maskimg = crack_area.Eachareas(merge_threshold=merge_threshold,area_threshold=area_threshold)
        total_area = sum(area.values())
        CrackNum = crack_area.Cracknums()
        label = Crack_label(CrackNum)
        cropped_cracks = Crop_cracks_img(mask, Bboxing)

        gujia_pos = Pixel_Coords(skeimg)
        classified_gujia_pos = Classify_points_basedon_Bbox(gujia_pos, Bboxing)
        crack_information=[]
        all_length=[]
        for i, crop in enumerate(cropped_cracks):
            crop_type = infertype(crop)
            new_classified_gujia_pos = reorder(classified_gujia_pos[label[i]], crop_type)
            spaceBetween, inflexion, connection = Crack_of_length(new_classified_gujia_pos)
            all_length.append(spaceBetween)
            crack_information.append([f"裂缝{label[i]},面积{area[label[i]]},长度{spaceBetween},拐点:{inflexion},连通:{connection}"])
            if showcrackinformation:
                print(f"{label[i]},裂缝面积为{area[label[i]]},裂缝长度:{spaceBetween},拐点:{inflexion},连通:{connection}")
            i += 1
        total_length = sum(all_length)
        if mode == 0:
            # 平均宽度
            width = Crack_of_width(total_area, total_length)
            return CrackNum, total_area, total_length, width, crack_information
        elif mode == 1:
            # 最大内切圆
            gray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            contours_arr, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            mask_gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            result, labelwidth = incircle(mask_gray, contours_arr)
            return CrackNum, total_area, total_length, labelwidth, crack_information

if __name__=="__main__":
    path = r"D:\pythonprojects\RoadCrack\dimension2_data\num\021.png"

    mask = cv2.imread(path)
    ske = Crack_skeleton()
    skeimg = ske.sketioncv(path)
    CrackNum, total_area, total_length, labelwidth, crack_information = DetectCracks(mask, skeimg, mode=1)

    print(f"裂缝个数为：{CrackNum}\n"
          f"裂缝总面积为：{total_area}\n"
          f"裂缝总长度为：{total_length}\n"
          f"裂缝宽度为：{labelwidth}\n"
          f"裂缝细致信息为：{crack_information}")