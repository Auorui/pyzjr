import cv2
import numpy as np
import pyzjr.Z as Z


__all__ = ["OverlayPng", "putBoxText", "cornerRect", "cvt2Center", "cvt2Corner", "AddText", "DrawPolygon", "DrawBboxPolygon"]


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
    text_x = text_y =None
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
        text_x = int(x1)
        text_y = int(y1 - text_size[1])
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

def cvt2Center(boxes):
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

def cvt2Corner(boxes):
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

def AddText(img, text, x, y, color, thickness=3):
    """
    Usage Directions:
        Using the OpenCV method to add text to images
    Code Example:
    ------------
    >>> img = np.zeros((200, 400, 3), dtype=np.uint8)
    >>> AddText(img, "Hello, OpenCV!", 50, 100, (0, 255, 0), 2)
    """
    cv2.putText(
        img,
        text,
        (x, y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        thickness=thickness,
        color=color,
        lineType=cv2.LINE_AA,
    )

def DrawPolygon(points, image, color):
    """
    Usage Directions:
        Draw a polygon onto an image using the given points and fill color.
    Code Example:
    ------------
    >>> image = np.ones((400, 400, 3), dtype=np.uint8)
    >>> polygon_points = np.array([[100, 100], [200, 50], [300, 150], [250, 300], [150, 250]])
    >>> result_image = DrawPolygon(polygon_points, image, (255, 0, 0))
    """
    points = np.array([points])
    points = points.astype(np.int32)
    image = cv2.fillPoly(image, points, color)
    return image

def DrawBboxPolygon(img, track_id, color, bbox, bboxcolor=(255, 0, 0)):
    """
    >>> bbox = np.array([100, 50, 300, 200])
    >>> result_image = DrawBboxPolygon(image, "Object 1", (0, 255, 0), bbox)
    """
    color = np.array(color)

    xmin, ymin, xmax, ymax = bbox.astype(np.int32).squeeze()

    bbox_h, bbox_w, _ = img[ymin:ymax, xmin:xmax].shape
    tiled_color = np.tile(color.reshape(1, 1, 3), (bbox_h, bbox_w, 1))
    img[ymin:ymax, xmin:xmax, :] = (img[ymin:ymax, xmin:xmax, :] + tiled_color) / 2.0

    white = (255, 255, 255)
    plot_x = xmin + 10
    plot_y = ymin + 25

    AddText(
        img,
        str(track_id),
        plot_x,
        plot_y,
        color=white,
        thickness=5,
    )
    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=bboxcolor)
    return img










