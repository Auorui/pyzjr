import cv2
import numpy as np
import pyzjr.Z as Z
from PIL import Image

def RemoveInboxes(boxes):
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

def cvtMask(pngpath, lower ,upper, convert=(255, 255, 255)):
    """
    去除标签中的杂色,并对其进行颜色的转换

    :param pngpath: 输入图像的文件路径
    :param lower: 颜色范围的下限值，作为一个包含(B, G, R)值的元组
    :param upper: 颜色范围的上限值，作为一个包含(B, G, R)值的元组
    :param convert: 需要转换为的颜色，作为一个包含(B, G, R)值的元组
    :return: 处理后的图像
    """
    image = cv2.imread(pngpath)
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)

    mask = cv2.inRange(image, lower, upper)
    image[mask > 0] = convert

    return image

def imageContentMove(image, H_pixels=None, V_pixels=None):
    if H_pixels is None:
        H_pixels = 0
    if V_pixels is None:
        V_pixels = 0
    h, w = image.shape[:2]
    black = np.zeros_like(image)

    start_row, end_row = max(0, V_pixels), min(h, h + V_pixels)
    start_col, end_col = max(0, H_pixels), min(w, w + H_pixels)

    fill_start_row, fill_end_row = max(0, -V_pixels), min(h, h - V_pixels)
    fill_start_col, fill_end_col = max(0, -H_pixels), min(w, w - H_pixels)

    black[start_row:end_row, start_col:end_col] = image[fill_start_row:fill_end_row, fill_start_col:fill_end_col]

    return black