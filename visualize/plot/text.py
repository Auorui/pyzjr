import cv2
import pyzjr.Z as Z

def AddText(img, text, x, y, color, thickness=1, font_scale=1,
            font_face=cv2.FONT_HERSHEY_SIMPLEX, line_type=cv2.LINE_AA, bottom_left_origin=False):
    """
    Adds text to an image at a specified location using OpenCV's `cv2.putText` method.

    Parameters:
    -----------
    img : numpy.ndarray
        The image (as a NumPy array) to which the text will be added.
    text : str
        The text string to be drawn on the image.
    x : int
        The x-coordinate of the bottom-left corner of the text.
    y : int
        The y-coordinate of the bottom-left corner of the text.
    color : tuple
        The color of the text in BGR format (e.g., (255, 0, 0) for blue).
    thickness : int, optional
        The thickness of the text lines (default is 1).
    font_scale : float, optional
        The scale factor for the font size (default is 1).
    font_face : int, optional
        The font type (default is `cv2.FONT_HERSHEY_SIMPLEX`).
    line_type : int, optional
        The type of line used to draw the text (default is `cv2.LINE_AA` for anti-aliased lines).
    bottom_left_origin : bool, optional
        If True, the origin of the text is at the bottom-left corner of the image.
        If False, the origin is at the top-left corner (default is False).

    Usage Example:
    --------------
    >>> import numpy as np
    >>> img = np.zeros((500, 800, 3), dtype=np.uint8)
    >>> AddText(img, "Hello, OpenCV!", 50, 100, (0, 255, 0), 2)
    """
    cv2.putText(
        img,
        text,
        (x, y),
        fontFace=font_face,
        fontScale=font_scale,
        thickness=thickness,
        color=color,
        lineType=line_type,
        bottomLeftOrigin=bottom_left_origin
    )
    return img

def PutMultiLineText(img, text, org, color, thickness=1, font_scale=1,
                     font_face=cv2.FONT_HERSHEY_SIMPLEX, line_type=cv2.LINE_8, bottom_left_origin=False):
    """
    Draws multi-line text on an image at a specified location, with support for line breaks (`\n`).

    Parameters:
    -----------
    img : numpy.ndarray
        The image (as a NumPy array) to which the text will be added.
    text : str
        The text string to be drawn on the image. Line breaks (`\n`) are supported.
    org : tuple
        The coordinates `(x, y)` of the bottom-left corner of the first line of text.
    color : tuple
        The color of the text in BGR format (e.g., (255, 0, 0) for blue).
    thickness : int, optional
        The thickness of the text lines (default is 1).
    font_scale : float, optional
        The scale factor for the font size (default is 1).
    font_face : int, optional
        The font type (default is `cv2.FONT_HERSHEY_SIMPLEX`).
    line_type : int, optional
        The type of line used to draw the text (default is `cv2.LINE_8`).
    bottom_left_origin : bool, optional
        If True, the origin of the text is at the bottom-left corner of the image.
        If False, the origin is at the top-left corner (default is False).

    Usage Example:
    --------------
    >>> import numpy as np
    >>> img = np.zeros((500, 800, 3), dtype=np.uint8)
    # >>> text = "Hello, OpenCV!\nThis is a test.\nCentered text with line breaks."
    >>> PutMultiLineText(img, text, (50, 50), (0, 255, 0), 1)
    """
    x, y = org
    text_lines = text.split('\n')
    # 获取文本行的高度（所有行的高度相同）
    _, line_height = cv2.getTextSize('', font_face, font_scale, thickness)[0]
    line_gap = line_height // 3

    for i, text_line in enumerate(text_lines):
        # 查找此行之前的文本的总大小
        line_y_adjustment = i * (line_gap + line_height)
        # 根据行号将文本从原始行向下移动
        if not bottom_left_origin:
            line_y = y + line_y_adjustment
        else:
            line_y = y - line_y_adjustment
        cv2.putText(img,
                    text=text_lines[i],
                    org=(x, line_y),
                    fontFace=font_face,
                    fontScale=font_scale,
                    color=color,
                    thickness=thickness,
                    lineType=line_type,
                    bottomLeftOrigin=bottom_left_origin)
    return img

def PutMultiLineCenteredText(img, text, color, thickness=1, font_scale=1,
                             font_face=cv2.FONT_HERSHEY_SIMPLEX, line_type=cv2.LINE_8):
    """
    Draws multi-line text centered both horizontally and vertically on an image.

    Parameters:
    -----------
    img : numpy.ndarray
        The image (as a NumPy array) to which the text will be added.
    text : str
        The text string to be drawn on the image. Line breaks (`\n`) are supported.
    color : tuple
        The color of the text in BGR format (e.g., (255, 0, 0) for blue).
    thickness : int, optional
        The thickness of the text lines (default is 1).
    font_scale : float, optional
        The scale factor for the font size (default is 1).
    font_face : int, optional
        The font type (default is `cv2.FONT_HERSHEY_SIMPLEX`).
    line_type : int, optional
        The type of line used to draw the text (default is `cv2.LINE_8`).

    Usage Example:
    --------------
    >>> import numpy as np
    >>> img = np.zeros((500, 800, 3), dtype=np.uint8)  # Create a blank black image
    # >>> text = "Hello, OpenCV!\nThis is a test.\nCentered text with line breaks."
    >>> PutMultiLineCenteredText(img, text, (0, 255, 0), 2)  # Add green centered multi-line text
    """
    img_h, img_w = img.shape[:2]
    text_lines = text.split('\n')
    # 获取文本行的高度（所有行的高度相同）
    _, line_height = cv2.getTextSize('', font_face, font_scale, thickness)[0]
    line_gap = line_height // 3

    text_block_height = len(text_lines) * (line_height + line_gap)
    text_block_height -= line_gap

    for i, text_line in enumerate(text_lines):
        line_width, _ = cv2.getTextSize(text_line, font_face, font_scale, thickness)[0]
        x = (img_w - line_width) // 2
        y = (img_h + line_height) // 2
        # 查找此行之前的文本的总大小
        line_adjustment = i * (line_gap + line_height)
        y += line_adjustment - text_block_height // 2 + line_gap

        cv2.putText(img,
                    text=text_lines[i],
                    org=(x, y),
                    fontFace=font_face,
                    fontScale=font_scale,
                    color=color,
                    thickness=thickness,
                    lineType=line_type)
    return img

def PutBoxText(img, bbox, text, text_pos_mode=-1,
               bboxcolor=Z.green, textcolor=Z.red, thickness=1, bboxthickness=1,
               font_scale=1, font_face=cv2.FONT_HERSHEY_SIMPLEX, line_type=cv2.LINE_8):
    """
    Draws a bounding box and text on an image. The text position can be customized relative to the bounding box.

    Parameters:
    -----------
    img : numpy.ndarray
        The image (as a NumPy array) on which the bounding box and text will be drawn.
    bbox : tuple
        The bounding box coordinates in the format `(x1, y1, x2, y2)`, where `(x1, y1)` is the top-left corner
        and `(x2, y2)` is the bottom-right corner.
    text : str
        The text string to be drawn on the image.
    text_pos_mode : int, optional
        Determines the position of the text relative to the bounding box:
        - `0`: Centered inside the bounding box.
        - `-1`: Top-left corner of the bounding box.
        - `1`: Top-right corner of the bounding box.
        Default is `0`.
    bboxcolor : tuple, optional
        The color of the bounding box in BGR format (e.g., (0, 255, 0) for green). Default is `Z.green`.
    textcolor : tuple, optional
        The color of the text in BGR format (e.g., (0, 0, 255) for red). Default is `Z.red`.
    thickness : int, optional
        The thickness of the text lines (default is 2).
    bboxthickness : int, optional
        The thickness of the bounding box lines (default is 2).
    font_scale : float, optional
        The scale factor for the font size (default is 1).
    font_face : int, optional
        The font type (default is `cv2.FONT_HERSHEY_SIMPLEX`).
    line_type : int, optional
        The type of line used to draw the text and bounding box (default is `cv2.LINE_8`).

    Usage Example:
    --------------
    >>> import numpy as np
    >>> img = np.zeros((500, 800, 3), dtype=np.uint8)
    >>> bbox = (150, 150, 300, 250)
    >>> PutBoxText(img, bbox, "Object", text_pos_mode=-1, bboxcolor=(0, 255, 0), textcolor=(0, 0, 255))
    """
    text_x = text_y = None
    x1, y1, x2, y2 = bbox
    text_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)
    cv2.rectangle(img, (x1, y1), (x2, y2), bboxcolor, bboxthickness)

    if text_pos_mode == 0:
        # Centered
        text_x = int((x1 + x2) / 2 - text_size[0] / 2)
        text_y = int((y1 + y2) / 2 + text_size[1] / 2)
    elif text_pos_mode == -1:
        # Top-left corner
        text_x = int(x1)
        text_y = int(y1 - text_size[1])
    elif text_pos_mode == 1:
        # Top-right corner
        text_x = int(x2 - text_size[0])
        text_y = int(y1 - text_size[1])

    cv2.putText(img,
                text=text,
                org=(text_x, text_y),
                fontFace=font_face,
                fontScale=font_scale,
                color=textcolor,
                thickness=thickness,
                lineType=line_type)
    return img

def PutRectangleText(img, text, bgcolor=(255, 255, 255), textcolor=(0, 0, 0), thickness=2,
                     font_scale=1.0, padding=10, font_face=cv2.FONT_HERSHEY_SIMPLEX,
                     line_type=cv2.LINE_8):
    """
    Draws a rectangle with a background color and places text inside it on an image.

    Parameters:
    -----------
    img : numpy.ndarray
        The image (as a NumPy array) on which the rectangle and text will be drawn.
    text : str
        The text string to be drawn inside the rectangle.
    bgcolor : tuple, optional
        The background color of the rectangle in BGR format (e.g., (255, 255, 255) for white).
    textcolor : tuple, optional
        The color of the text in BGR format (e.g., (0, 0, 0) for black). Default is `Z.black`.
    thickness : int, optional
        The thickness of the text lines (default is 2).
    font_scale : float, optional
        The scale factor for the font size (default is 1.0).
    padding : int, optional
        The padding (in pixels) between the text and the edges of the rectangle (default is 10).
    font_face : int, optional
        The font type (default is `cv2.FONT_HERSHEY_SIMPLEX`).
    line_type : int, optional
        The type of line used to draw the text and rectangle (default is `cv2.LINE_8`).

    Usage Example:
    --------------
    >>> import numpy as np
    >>> img = np.zeros((500, 800, 3), dtype=np.uint8)  # Create a blank black image
    >>> PutRectangleText(img, "Hello, World!", bgcolor=(255, 255, 255), textcolor=(0, 0, 0))
    """
    (text_width, text_height), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
    x1 = y1 = padding
    x2 = x1 + text_width + 2 * padding
    y2 = y1 + text_height + 2 * padding
    cv2.rectangle(img, (x1, y1), (x2, y2), bgcolor, -1)

    cv2.putText(img,
                text=text,
                org=(x1 + padding, y1 + text_height + padding),
                fontFace=font_face,
                fontScale=font_scale,
                color=textcolor,
                thickness=thickness,
                lineType=line_type)

    return img


if __name__=="__main__":
    import numpy as np
    img = np.zeros((500, 800, 3), dtype=np.uint8)
    text = "Hello, OpenCV!\nThis is a test.\nCentered text with line breaks."
    PutMultiLineText(img, text, (50, 50), (0, 255, 0), 1)
    PutMultiLineCenteredText(img, text, (0, 255, 255), 1)
    bbox = (150, 150, 300, 250)
    PutBoxText(img, bbox, "Object", text_pos_mode=-1, bboxcolor=(0, 255, 0), textcolor=(0, 0, 255))
    PutRectangleText(img, "Hello, World!", bgcolor=(255, 255, 255), textcolor=(0, 0, 0))
    cv2.imshow("test", img)
    cv2.waitKey(0)