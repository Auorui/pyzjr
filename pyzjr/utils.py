import cv2
import numpy as np
import pyzjr.Z as Z
from PIL import Image
import torch

def gpu(i=0,cuda=True):
    """
    如果cuda为True并且gpu可用,则返回gpu(i),否则返回cpu().
    :param i: 索引i,表示使用第几块gpu
    :param cuda: 布尔值,表示是否使用cuda,没有可以使用cpu,设置为False
    """
    if cuda and torch.cuda.is_available() and torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def allgpu():
    """
    Those who can use this function must be very rich.
    """
    devices = []
    for i in range(torch.cuda.device_count()):
        devices.append(i)
    return devices if devices else [torch.device('cpu')]

def addnoisy(image, n=10000):
    """
    :param image: 原始图像
    :param n: 添加椒盐的次数,默认为10000
    :return: 返回被椒盐处理后的图像
    """
    result = image.copy()
    w, h = image.shape[:2]
    for i in range(n):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        if np.random.randint(0, 2) == 0:
            result[x, y] = 0
        else:
            result[x, y] = 255
    return result

class Boundingbox():
    def RemoveInboxes(self,boxes):
        """
        中心点的方法,去除重叠的框,并去掉相对较小的框
        :param boxes: [minc, minr, maxc, maxr]
        :return: 过滤的boxes
        """
        final_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            box_area = (x2 - x1) * (y2 - y1)
            is_inner = False
            for fb in final_boxes:
                fx1, fy1, fx2, fy2 = fb
                if fx1 <= cx <= fx2 and fy1 <= cy <= fy2:
                    fb_area = (fx2 - fx1) * (fy2 - fy1)
                    if box_area >= fb_area:
                        final_boxes.remove(fb)
                    else:
                        is_inner = True
                        break
            if not is_inner:
                final_boxes.append(box)
        return final_boxes

    def cvt2Center(self,boxes):
        """
        Convert from (upper left, lower right) to (center, width, height).
        :param boxes: [x1, y1, x2, y2]
        :return: [cx, cy, w, h]
        """
        converted_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            converted_box = [cx, cy, w, h]
            converted_boxes.append(converted_box)
        return converted_boxes

    def cvt2Corner(self,boxes):
        """Convert from (center, width, height) to (upper left, lower right).
        :param boxes: [cx, cy, w, h]
        :return: [x1, y1, x2, y2]
        """
        converted_boxes = []
        for box in boxes:
            cx, cy, w, h = box[0], box[1], box[2], box[3]
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            converted_box = [x1, y1, x2, y2]
            converted_boxes.append(converted_box)
        return converted_boxes

    def putBoxText(self,background_image, bbox, text, mode=1, rect=False, bboxcolor=(0, 255, 0), textcolor=(0, 0, 255),
                   fontsize=1, thickness=2, bboxthickness=2,font=cv2.FONT_HERSHEY_SIMPLEX):
        """
        在给定的背景图像上绘制一个框，并在框的中心位置添加文本。获取文本的text_size，使文本居中。

        :param background_image: 背景图像，要在其上绘制框和文本的图像
        :param bbox: 框的边界框，表示为 [x1, y1, x2, y2]
        :param text: 要添加到框中心的文本
        :param mode: 模式0 表示框的中心位置，1 表示框的左上角
        :param rect: 是否绘制角标记框，导入的为 cornerRect 函数
        :param bboxcolor: 框的颜色，以 BGR 格式表示，例如 (0, 255, 0) 表示绿色
        :param textcolor: 文本的颜色，以 BGR 格式表示，例如 (0, 0, 255) 表示红色
        :param fontsize: 文本的字体大小
        :param thickness: 文本的线宽
        :param bboxthickness: 框的线宽，默认为 2
        :return: 绘制了框和文本的图像
        """
        x1, y1, x2, y2 = bbox
        text_size, _ = cv2.getTextSize(text, font, fontsize, thickness)
        if rect:
            background_image = self.cornerRect(background_image, bbox)
        else:
            cv2.rectangle(background_image, (x1, y1), (x2, y2), bboxcolor, bboxthickness)
        if mode == 0:
            text_x = int((x1 + x2) / 2 - text_size[0] / 2)
            text_y = int((y1 + y2) / 2 + text_size[1] / 2)
        elif mode == 1:
            text_x = x1
            text_y = y1 - text_size[1]
        cv2.putText(background_image, text, (text_x, text_y), font, fontsize, textcolor, thickness)

    def cornerRect(self,drawimg, bbox, length=30, lthickness=5, rthickness=1, bboxcolor=Z.purple, colorCorner=Z.green):
        """
        在图像上绘制带有角标记的矩形框
        :param drawimg: 需要绘制的图像
        :param bbox: 边界框, 表示为 [x, y, x1, y1]
        :param length: 角标记的长度
        :param lthickness: 角标记的线条宽度
        :param rthickness: 矩形框的线条宽度
        :param bboxcolor: 矩形框的颜色
        :param colorCorner: 角标记的颜色
        :return: 绘制了角标记的图像副本
        """
        x, y, x1, y1 = bbox
        cv2.rectangle(drawimg, (x, y), (x1, y1), bboxcolor, rthickness)
        # 左上角  (x,y)
        cv2.line(drawimg, (x, y), (x + length, y), colorCorner, lthickness)
        cv2.line(drawimg, (x, y), (x, y + length), colorCorner, lthickness)
        # 右上角  (x1,y)
        cv2.line(drawimg, (x1, y), (x1 - length, y), colorCorner, lthickness)
        cv2.line(drawimg, (x1, y), (x1, y + length), colorCorner, lthickness)
        # 左下角  (x,y1)
        cv2.line(drawimg, (x, y1), (x + length, y1), colorCorner, lthickness)
        cv2.line(drawimg, (x, y1), (x, y1 - length), colorCorner, lthickness)
        # 右下角  (x1,y1)
        cv2.line(drawimg, (x1, y1), (x1 - length, y1), colorCorner, lthickness)
        cv2.line(drawimg, (x1, y1), (x1, y1 - length), colorCorner, lthickness)

        return drawimg

def cvt8png(pngpath, bit_depth=False, target=Z.white, convert=(128, 0, 0)):
    """
    Voc: RGB png彩图转换
    :param bit_depth: 默认转为8位,需要使用out_png.save可以正确保存
    True:                      False:
        plt.imshow(img)             cv2.imshow("img",img)
        plt.axis('off')             cv2.waitKey(0)
        plt.show()
    :param pngpath: 为保证是按照cv2的方式读入,所以传入路径即可
    :param target: 目标颜色,RGB方式
    :param convert: 转换颜色,同RGB格式
    :return:
    """
    png = cv2.imread(pngpath)
    png = cv2.cvtColor(png, cv2.COLOR_BGR2RGB)
    h, w = png.shape[:2]

    mask = np.all(png == target, axis=-1)
    out_png = np.zeros((h, w, 3), dtype=np.uint8)
    out_png[mask] = convert
    out_png[~mask] = png[~mask]

    if bit_depth:
        out_png_pil = Image.fromarray(out_png)
        out_png_pil = out_png_pil.convert("P", palette=Image.ADAPTIVE, colors=256)
        return out_png_pil
    else:
        out_png_cv = cv2.cvtColor(out_png, cv2.COLOR_RGB2BGR)
        return out_png_cv

def empty(a):
    pass

