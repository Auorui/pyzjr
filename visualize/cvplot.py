"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is used for image drawing.
"""
import cv2
import numpy as np
import pyzjr.Z as Z

__all__ = ["OverlayPng", "PutBoxText", "AddText", "ConvertBbox", "DrawPolygon", "CornerRect"]

def OverlayPng(imgBack, imgFront, pos=(0, 0), alpha_gain=1.0):
    """
    Overlay display image with proper alpha blending
    :param imgBack: Background image, no format requirement, 3 channels
    :param imgFront: PNG pre image, must be read using cv2.IMREAD_UNCHANGED=-1
    :param pos: Placement position

    Examples:
    '''
        background = cv2.imread(background_path)
        overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        fused_image = pyzjr.OverlayPng(background, overlay, alpha_gain=1.5)
    '''
    """
    hf, wf, cf = imgFront.shape
    hb, wb, cb = imgBack.shape
    y_pos, x_pos = pos
    y_end = y_pos + hf
    x_end = x_pos + wf

    # Ensure we don't go beyond the background boundaries
    if y_end > hb:
        y_end = hb
    if x_end > wb:
        x_end = wb

    # Resize overlay to fit the background (optional but good practice)
    overlay_resized = cv2.resize(imgFront, (x_end - x_pos, y_end - y_pos))

    overlay_alpha = overlay_resized[:, :, 3].astype(float) / 255.0
    overlay_alpha = np.clip(overlay_alpha * alpha_gain, 0, 1)
    background_alpha = 1.0 - overlay_alpha

    result = overlay_resized[:, :, :3] * overlay_alpha[..., np.newaxis] + imgBack[y_pos:y_end, x_pos:x_end, :3] * background_alpha[..., np.newaxis]

    imgBack[y_pos:y_end, x_pos:x_end, :3] = result.astype(np.uint8)

    return imgBack


def PutBoxText(imgBack, bbox, text, mode=0, bboxcolor=Z.green, textcolor=Z.red,
               fontsize=.5, thickness=2, bboxthickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Add text within a bounding box on an image.

    Args:
        imgBack (numpy.ndarray): The background image where the text and bounding box will be drawn.
        bbox (list of int): The bounding box coordinates in the format [x1, y1, x2, y2].
        text (str): The text to be drawn.
        mode (int): Determines the position of the text within the bounding box.
            - -1: Top-left corner.
            - 0: Centered.
            - 1: Top-right corner.
        bboxcolor: The RGB color of the bounding box.
        textcolor: The RGB color of the text.
        fontsize (float): Font size of the text.
        thickness (int): Thickness of the text.
        bboxthickness (int): Thickness of the bounding box.
        font (int): Font type to be used.
    """
    text_x = text_y = None
    x1, y1, x2, y2 = bbox
    text_size, _ = cv2.getTextSize(text, font, fontsize, thickness)
    cv2.rectangle(imgBack, (x1, y1), (x2, y2), bboxcolor, bboxthickness)

    if mode == 0:
        # Centered
        text_x = int((x1 + x2) / 2 - text_size[0] / 2)
        text_y = int((y1 + y2) / 2 + text_size[1] / 2)
    elif mode == -1:
        # Top-left corner
        text_x = int(x1)
        text_y = int(y1 - text_size[1])
    elif mode == 1:
        # Top-right corner
        text_x = int(x2 - text_size[0])
        text_y = int(y1 - text_size[1])

    cv2.putText(imgBack, text, (text_x, text_y), font, fontsize, textcolor, thickness)


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


def ConvertBbox(bbox, to_center=True):
    """
    Convert bounding box coordinates between (upper left, lower right) and (center, width, height).

    :param bbox: List of bounding boxes in either [x1, y1, x2, y2] or [cx, cy, w, h] format.
    :param to_center: If True, convert from (upper left, lower right) to (center, width, height).
                      If False, convert from (center, width, height) to (upper left, lower right).
    :return: List of converted bounding boxes.
    """
    converted_boxes = []
    for box in bbox:
        if to_center:
            # Convert from (upper left, lower right) to (center, width, height)
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            converted_box = [cx, cy, w, h]
        else:
            # Convert from (center, width, height) to (upper left, lower right)
            cx, cy, w, h = box
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            converted_box = [x1, y1, x2, y2]
        converted_boxes.append(converted_box)
    return converted_boxes


def CornerRect(drawimg, bbox, length=30, lthickness=5, rthickness=1,
               bboxcolor=Z.purple, colorCorner=Z.green):
    """
    Draw a rectangle with corner marks on an image
    :param drawimg: The image to draw on
    :param bbox: The bounding box, represented as [x1, y1, x2, y2]
    :param length: The length of the corner marks
    :param lthickness: The line thickness of the corner marks
    :param rthickness: The line thickness of the rectangle
    :param bboxcolor: The color of the rectangle
    :param colorCorner: The color of the corner marks
    :return: A copy of the image with corner marks drawn
    """
    x, y, x1, y1 = bbox
    cv2.rectangle(drawimg, (x, y), (x1, y1), bboxcolor, rthickness)
    # Top-left corner (x, y)
    cv2.line(drawimg, (x, y), (x + length, y), colorCorner, lthickness)
    cv2.line(drawimg, (x, y), (x, y + length), colorCorner, lthickness)
    # Top-right corner (x1, y)
    cv2.line(drawimg, (x1, y), (x1 - length, y), colorCorner, lthickness)
    cv2.line(drawimg, (x1, y), (x1, y + length), colorCorner, lthickness)
    # Bottom-left corner (x, y1)
    cv2.line(drawimg, (x, y1), (x + length, y1), colorCorner, lthickness)
    cv2.line(drawimg, (x, y1), (x, y1 - length), colorCorner, lthickness)
    # Bottom-right corner (x1, y1)
    cv2.line(drawimg, (x1, y1), (x1 - length, y1), colorCorner, lthickness)
    cv2.line(drawimg, (x1, y1), (x1, y1 - length), colorCorner, lthickness)

if __name__=="__main__":
    import cv2
    imagePath = r"D:\PythonProject\pyzjr\pyzjr\test.png"
    image = np.ones((400, 400, 3), dtype=np.uint8)

    img = cv2.imread(imagePath)
    CornerRect(img, [50, 50, 200, 200])
    # PutBoxText(img, [50, 50, 200, 200], text="DOG", mode=0)
    cv2.imshow("show", img)
    cv2.waitKey(0)
