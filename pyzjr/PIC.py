import cv2
from pylab import *
from pyzjr.augmentation import ColorFind
import pyzjr.Z as Z
from pyzjr.math import *
from skimage.morphology import disk
from skimage.filters import rank
from skimage import measure
from torchvision import transforms

def repairImg(img, hsvval, r=5,flags=Z.repair_NS):
    """
    * 用于修复图像
    :param img: 输入图像
    :param hsvval: hsv值,[[hmin, smin, vmin], [hmax, smax, vma]]
    :param r: 修复半径，即掩膜的像素周围需要参考的区域半径
    :param flags: 修复算法的标志,有Z.repair_NS、Z.repair_TELEA,默认为Z.repair_NS
    :param mode: 是否采用HSV例模式,默认为0,自定义模式,可通过Color下的TrackBar文件中获得
    :return: 返回修复后的图片,以及检测到物体
    """
    ColF=ColorFind()
    imgResult, mask = ColF.MaskZone(img, hsvval)
    dst = cv2.inpaint(img, mask, r, flags)
    return dst, imgResult


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


def labelpoint(im, click=4):
    """
    交互式标注
    :param im: 图像,采用im = Image.open(?)的方式打开，颜色空间才是正常的
    :param click: 点击次数、默认为4
    :return: 返回点的坐标
    """
    imshow(im)
    print(f'please click {click} points')
    x=ginput(click)
    print('you clicked:',x)
    return x


def transImg(img,targetWH):
    """
    标注方式按照先上后下，先左后右的顺序
    :param img: 图像
    :param targetWH: 已知目标物体的宽度,长度
    :return: 修正后的图像
    """
    width, height = targetWH
    pts1 = np.float32(labelpoint(img))
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))
    return imgOutput


def BinaryImg(img, min_value=127, max_value=255):
    """
    将OpenCV读取的图像转换为二值图像
    :param img: 输入的图像,必须是OpenCV读取的图像对象(BGR格式)
    :param min_value: 最小阈值
    :param max_value: 最大阈值
    :return: 转换后的二值图像
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, min_value, max_value, cv2.THRESH_BINARY)
    return binary_image


def SearchOutline(img):
    """
    在给定图像中搜索轮廓并返回轮廓的坐标列表。
    :param img: 要搜索轮廓的图像。
    :return: 包含轮廓坐标的列表，每个坐标表示裂缝的一个点，坐标格式为 [(x, y),...]。
    """
    binary_image = BinaryImg(img)
    contours = measure.find_contours(binary_image, level=128, fully_connected='low', positive_orientation='low')
    contour_coordinates = []
    for contour in contours:
        contour_coordinates.extend([(int(coord[1]), int(coord[0])) for coord in contour])
    return contour_coordinates


def imtensor(image, dim=0):
    """转化为tensor格式
    image1 = Image.open(path).convert('L')
    image2 = pz.imtensor(image1)
    """
    preprocess = transforms.ToTensor()
    image_tensor = preprocess(image).unsqueeze(dim=dim)
    return image_tensor

def torch2pillow(input_tensor, dim=0):
    """将tensor再转化为PIL"""
    output_image = transforms.ToPILImage()(input_tensor.squeeze(dim=dim))
    return output_image

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

def RectMask(image,mask,boxes):
    """
    创建矩形蒙版
    :param image: 原始图像。
    :param mask: np.zeros(image.shape, dtype=np.uint8) 背景图
    :param boxes: [x1, y1, x2, y2]
    :return: 蒙版与应用蒙版
    """
    for box in boxes:
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 255
    if mask.ndim == 3 and mask.shape[-1] > 1:
        mask = mask[:, :, 0]
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return mask, masked_image


def drawOutline(blackbackground,contours,mode=0,color=Z.green):
    """
    绘制边缘轮廓
    :param blackbackground: 背景图
    :param contours: [(x,y),...]
    :param mode: 默认模式0,可选模式0或1
    :param color: 默认绿色
    :return: 
    """
    if mode==0:
        contours = Advancedlist(contours, "Y")
        contour = [np.array([point], dtype=np.int32) for point in contours]
        cv2.drawContours(blackbackground, contour, -1, color, thickness=-1)
    elif mode==1:
        for i in range(len(contours) - 1):
            start = contours[i]
            end = contours[i + 1]
            cv2.line(blackbackground, start, end, color=color, thickness=1)


def gradientOutline(img,radius=2):
    """
    对图像进行梯度边缘检测，并返回边缘强度图像。
    
    :param img: 输入的图像(内部有二值转化)
    :param radius: 半径,默认为2。
    :return: 返回经过梯度边缘检测处理后的边缘强度图像
    """
    image = BinaryImg(img)
    denoised = rank.median(image, disk(radius))
    gradient = rank.gradient(denoised, disk(radius))
    gradient = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

    return gradient


def OverlayPng(imgBack, imgFront, pos=(0, 0)):
    """
    叠加显示图片
    :param imgBack: 背景图像,无格式要求,3通道
    :param imgFront: png前置图片,读取方式必须使用 cv2.IMREAD_UNCHANGED = -1
    :param pos: 摆放位置
    :return: 叠加显示的图片
    """
    hf, wf, cf = imgFront.shape
    hb, wb, cb = imgBack.shape
    y_pos, x_pos = pos
    y_end = y_pos + hf
    x_end = x_pos + wf
    if y_end > hb:
        y_end = hb
    if x_end > wb:
        x_end = wb
    overlay_region = imgFront[:y_end - y_pos, :x_end - x_pos, :]
    overlay_mask = overlay_region[:, :, 3]
    overlay_alpha = overlay_mask.astype(float) / 255.0
    background_alpha = 1.0 - overlay_alpha
    result = overlay_region[:, :, :3] * overlay_alpha[..., np.newaxis] + imgBack[y_pos:y_end, x_pos:x_end, :3] * background_alpha[..., np.newaxis]
    imgBack[y_pos:y_end, x_pos:x_end, :3] = result

    return imgBack
