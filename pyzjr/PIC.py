import cv2
from pylab import *
from pyzjr.ColorModule import ColorFind
import pyzjr.Z as Z
from pyzjr.zmath import *
from skimage.morphology import skeletonize,disk
from skimage.filters import threshold_otsu,rank
from skimage.color import rgb2gray
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
    :return: 返回修复后的图片
    """
    ColF=ColorFind.getMask()
    imgResult, mask, imgHSV = ColF.MaskZone(img, hsvval)
    dst = cv2.inpaint(img, mask, r, flags)
    return dst

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


def SkeletonMap(crack):
    """
    获取骨架图的信息
    :param crack: 目标图
    :return: 骨架图与一个数组，其中每一行表示一个非零元素的索引(y,x)，包括行索引和列索引
    """
    gray = rgb2gray(crack)
    thresh = threshold_otsu(gray)
    binary = gray > thresh
    skeimage = skeletonize(binary)
    skepoints = np.argwhere(skeimage)
    return skeimage, skepoints

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


def putBoxText(background_image, bbox, text, mode=1, rect=False, bboxcolor=(0, 255, 0), textcolor=(0, 0, 255),
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
        background_image = cornerRect(background_image, bbox)
    else:
        cv2.rectangle(background_image, (x1, y1), (x2, y2), bboxcolor, bboxthickness)
    if mode == 0:
        text_x = int((x1 + x2) / 2 - text_size[0] / 2)
        text_y = int((y1 + y2) / 2 + text_size[1] / 2)
    elif mode == 1:
        text_x = x1
        text_y = y1 - text_size[1]
    cv2.putText(background_image, text, (text_x, text_y), font, fontsize, textcolor, thickness)

def cornerRect(drawimg, bbox, length=30, lthickness=5, rthickness=1, bboxcolor=Z.purple, colorCorner=Z.green):
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

def Searching_contours(img):
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


def cvtColor(image):
    """转化为RGB格式"""
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        img = image.convert('RGB')
        return img

def imtensor(image,dim=0):
    """转化为tensor格式
    image1 = Image.open(path).convert('L')
    image2 = pz.imtensor(image1)
    """
    preprocess = transforms.ToTensor()
    image_tensor = preprocess(image).unsqueeze(dim=dim)
    return image_tensor

def impillow(input_tensor,dim=0):
    """将tensor再转化为PIL"""
    output_image = transforms.ToPILImage()(input_tensor.squeeze(dim=dim))
    return output_image

def RectMask(image, boxes):
    """
    创建矩形蒙版
    :param image: 原始图像。
    :param boxes: [x1, y1, x2, y2]
    :return: 蒙版与应用蒙版
    """
    mask = np.zeros(image.shape, dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 255
    if mask.ndim == 3 and mask.shape[-1] > 1:
        mask = mask[:, :, 0]
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return mask, masked_image


def BilinearImg(image, scale):
    """
    双线性插值
    :param image: 原始图像。
    :param scale: 规格,如1.5为,放大1.5倍
    """
    ah, aw, channel = image.shape
    bh, bw = int(ah * scale), int(aw * scale)
    dst_img = np.zeros((bh, bw, channel), np.uint8)

    y_coords, x_coords = np.meshgrid(np.arange(bh), np.arange(bw), indexing='ij')
    AX = (x_coords + 0.5) / scale - 0.5   # 移向像素中心
    AY = (y_coords + 0.5) / scale - 0.5

    x1 = np.floor(AX).astype(int)
    y1 = np.floor(AY).astype(int)
    x2 = np.minimum(x1 + 1, aw - 1)
    y2 = np.minimum(y1 + 1, ah - 1)
    R1 = ((x2 - AX)[:, :, np.newaxis] * image[y1, x1]).astype(float) + (
                (AX - x1)[:, :, np.newaxis] * image[y1, x2]).astype(float)
    R2 = ((x2 - AX)[:, :, np.newaxis] * image[y2, x1]).astype(float) + (
                (AX - x1)[:, :, np.newaxis] * image[y2, x2]).astype(float)

    dst_img = (y2 - AY)[:, :, np.newaxis] * R1 + (AY - y1)[:, :, np.newaxis] * R2

    return dst_img.astype(np.uint8)

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
    :param imgBack: 无格式要求,3通道
    :param imgFront: png图片,读取方式 cv2.IMREAD_UNCHANGED = -1
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

