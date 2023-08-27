import cv2
import numpy as np
import pyzjr.Z as Z
from PIL import Image

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

def getContours(img, cThr=(100, 100), minArea=1000, filter=0, draw=True):
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

def empty(a):
    pass

def Leftclick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked on image coordinates: ({x}, {y})")
        param[0] -= 1
        if param[0] >= 0:
            param[2].append((x, y))
        if param[0] == 0:
            cv2.setMouseCallback(param[1], lambda *args: None)

def Mousecallback(click=3, name="Mark Points", destroy=False):
    points = [click, name, []]
    cv2.namedWindow(name)
    cv2.setMouseCallback(name, Leftclick, points)
    print("Please click", click, "points")

    while points[0] > 0:
        cv2.waitKey(10)
    if destroy:
        cv2.destroyAllWindows()
    return points[2]