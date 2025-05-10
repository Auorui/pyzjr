import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import morphology, measure

from pyzjr.visualize.plot.text import PutBoxText
from pyzjr.augmentation.mask.predeal import binarization, bool2mask

# 裂缝的类型
class CrackType():
    """直方图投影法推断裂缝类型, 能区分线性裂缝和网状裂缝"""
    def __init__(self, threshold=3, HWratio=10, Histratio=0.5, alpha=15, beta=55):
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

    def inference_minAreaRect(self, mask):
        """
        从二值化掩膜图像中获取最小外接矩形的相关信息, 旋转矩形框长边与x轴的夹角.
        旋转角度 angle 是相对于图像水平方向的夹角，范围是 -90 到 +90 度.
        然而，一般情况下，我们习惯将角度定义为相对于 x 轴正方向的夹角，范围是 -180 到 +180 度.
        """
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_merge = np.vstack(contours)
        minAreaRect = cv2.minAreaRect(contour_merge)
        w, h = minAreaRect[1]
        if w > h:
            angle = int(minAreaRect[2])
        else:
            angle = -(90 - int(minAreaRect[2]))

        return w, h, angle

    def hist_judge(self, hist_v, esp=1e-5):
        less_than_T = np.count_nonzero((hist_v > 0) & (hist_v <= self.threshold))
        more_than_T = np.count_nonzero(hist_v > self.threshold)
        return more_than_T / (less_than_T + esp) > self.Histratio

    def angle2cls(self, angle):
        angle = abs(angle)
        assert 0 <= angle <= 90, "ERROR: The angle value exceeds the limit and should be between 0 and 90 degrees!"
        if angle < self.alpha:
            return self.types[0]
        elif self.alpha <= angle <= self.beta:
            return self.types[2]
        elif angle > self.beta:
            return self.types[1]
        else:
            return None

    def skeleton_map(self, binary):
        """获取骨架图的信息, 因为只是用于判断类型, 所以不要求其是否有毛刺"""
        skimage = skeletonize(binary)
        skimage = morphology.closing(skimage, morphology.disk(3))
        skepoints = np.argwhere(skimage)
        skimage = skimage.astype(np.uint8) * 255
        return skimage, skepoints

    def classify(self, mask):
        """
        针对当前crack instance，对其进行分类；
        主要利用了骨骼点双向投影直方图、旋转矩形框宽高比 /角度；
        """
        H, W = mask.shape[:2]
        mask = binarization(mask)
        skeimage, skeleton_pts = self.skeleton_map(mask)
        w, h, angle = self.inference_minAreaRect(skeimage)
        if w / h < self.HWratio or h / w < self.HWratio:
            pts_y, pts_x = skeleton_pts[:, 0], skeleton_pts[:, 1]
            hist_x = np.histogram(pts_x, W)
            hist_y = np.histogram(pts_y, H)
            if self.hist_judge(hist_x[0]) and self.hist_judge(hist_y[0]):
                return self.types[3]

        return self.angle2cls(angle)


class detect_crack():
    def __init__(self, mask):
        self.mask_copy = mask.copy()
        self.crack_type = CrackType()
        self.overall_crack_category = self.crack_type.classify(mask)  # 判断整体类型
        self.total_length = 0
        self.max_wide = 0
        connected_thresh = self.preprocess_binary(mask)
        self.total_area = np.count_nonzero(connected_thresh)
        self.regions_info = self.get_label_region(connected_thresh)
        self.crack_summary = {
            "crack_category": self.overall_crack_category,  # 整体裂缝类别
            "total_area": self.total_area,  # 总面积
            "crack_number": len(self.regions_info),  # 裂缝个数
            "crack_length": self.total_length / 2,
            "crack_max_wide": self.max_wide
        }
        # 将每个区域的信息添加到字典中
        for i, region in enumerate(self.regions_info):
            region_key = f"Region{i+1}"
            self.crack_summary[region_key] = region

    def preprocess_binary(self, mask):
        # 二值化处理
        thresh = binarization(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # 将相邻点连接在一起，避免因为单个像素间断出现多条裂缝
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return connected

    def get_label_region(self, thresh, connectivity=2):
        # 获取标记的区域及其额外属性
        def region_incircle(regionmask, intensity_image=None):
            # 计算区域的内切圆直径
            max_val = 0
            max_dist_pt = None
            regionmask = bool2mask(regionmask)
            dist_transform = cv2.distanceTransform(regionmask, cv2.DIST_L2, 3)
            min_val, curr_max_val, _, curr_max_dist_pt = cv2.minMaxLoc(dist_transform)
            if curr_max_val > max_val:
                max_val = curr_max_val
                max_dist_pt = curr_max_dist_pt
            wide = max_val * 2
            return wide, max_dist_pt
        labeled_image = measure.label(thresh, connectivity=connectivity)
        region_props = measure.regionprops(labeled_image, extra_properties=(region_incircle,))
        regions_info = []
        for i, region in enumerate(region_props):
            region_info = {}
            region_info["crack_category"] = self.crack_type.angle2cls(90 - abs(np.degrees(region.orientation)))  # 根据方向计算裂缝类型
            region_info["area"] = region.area  # 记录区域的面积
            region_info["perimeter"] = region.perimeter  # 记录区域的周长
            region_info["orientation"] = 90 - abs(np.degrees(region.orientation))  # 记录区域的方向
            region_info["max_width"] = region.region_incircle[0]  # 记录内切圆的最大宽度
            minr, minc, maxr, maxc = region.bbox
            region_info["max_width_point"] = (
                region.region_incircle[1][0]+minc, region.region_incircle[1][1]+minr
            )  # 记录内切圆最大宽度的点
            region_info["bounding_box"] = [(minc, minr), (maxc, maxr)]  # 记录区域的边界框
            regions_info.append(region_info)
            self.total_length += region.perimeter
            if region.region_incircle[0] > self.max_wide:
                self.max_wide = region.region_incircle[0]

        return regions_info

    def summary(self):
        def format_summary(data, indent=0):
            """递归格式化字典为结构化的字符串"""
            formatted_str = ""
            for key, value in data.items():
                if isinstance(value, dict):
                    formatted_str += " " * indent + f"{key}:\n"
                    formatted_str += format_summary(value, indent + 4)
                else:
                    formatted_str += " " * indent + f"{key}: {value}\n"
            return formatted_str

        return format_summary(self.crack_summary)


    def plot(self, circlecolor=(0, 255, 0), bboxcolor=(255, 0, 0), textcolor=(0, 0, 255),
             font_scale=.7, thickness=1):
        """
        绘制带有内切圆、边界框和标注的区域图像
        :param circle_color: 内切圆的颜色 (默认绿色)
        :param bboxcolor: 边界框的颜色 (默认红色)
        :param textcolor: 标注文字的颜色 (默认黄色)
        :param font_scale: 标注文字的大小
        :param thickness: 文字与边界框的厚度
        :return: 绘制后的图像
        """
        plot_image = self.mask_copy.copy()
        for i, region in enumerate(self.regions_info):
            # 画内切圆
            max_width = region["max_width"]
            max_width_point = region["max_width_point"]
            cv2.circle(plot_image, max_width_point, int(max_width / 2), circlecolor, 1, 1, 0)
            # 画边界框, 标注区域类别
            bbox = region["bounding_box"]
            label = f"Region {i+1}"
            PutBoxText(plot_image, (bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]), label, 0, bboxcolor, textcolor,
                       font_scale=font_scale, thickness=thickness, bboxthickness=thickness)
        return plot_image

if __name__=="__main__":
    img_mask = cv2.imread(r"E:\PythonProject\pyzjrPyPi\pyzjr\measure\fissure\0028.png")
    # crack_type = CrackType()
    # result = crack_type.classify(img_mask)
    decrack = detect_crack(img_mask)
    print(decrack.summary())
    plot_image = decrack.plot()
    # print(result)
    # cv2.imwrite("decrack.png", plot_image)
    cv2.imshow("Crack Detection", plot_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()