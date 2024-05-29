import cv2
from pylab import *
import pyzjr.Z as Z
from pyzjr.Math import *
from skimage.morphology import disk
from skimage.filters import rank
from skimage import measure
from pyzjr.augmentation.mask_ops import BinaryImg
from PIL import Image

def convert_pil_to_cv(pil_image):
    """将PIL图像转换为OpenCV图像"""
    if pil_image.mode == 'L':
        open_cv_image = np.array(pil_image)
    else:
        open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return open_cv_image

def convert_cv_to_pil(cv_image):
    """将OpenCV图像转换为PIL图像"""
    if cv_image.ndim == 2:
        pil_image = Image.fromarray(cv_image, mode='L')
    else:
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    return pil_image

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
    x = ginput(click)
    print('you clicked:',x)
    return x

def get_warp_perspective(img, targetWH):
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
    image = BinaryImg(img)
    denoised = rank.median(image, disk(radius))
    gradient = rank.gradient(denoised, disk(radius))
    gradient = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

    return gradient
