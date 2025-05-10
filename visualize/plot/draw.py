import cv2
import numpy as np
import pyzjr.Z as Z

def DrawPolygon(img, points, color):
    """
    Draw a polygon onto an image using the given points and fill color.
    Parameters:
    -----------
    img : numpy.ndarray
        The image (as a NumPy array) on which the rectangle and corners will be drawn.
    points : list of tuples or numpy.ndarray
            List of vertex coordinates for polygons. Each coordinate should be a tuple (x, y) containing two integers,
            Alternatively, a NumPy array with the shape (n, 1, 2) can be passed, where n is the number of vertices of the polygon.
    color : tuple
        Fill the color of the polygon in the format of (B, G, R), with each value ranging from 0 to 255 (inclusive)
    Code Example:
    ------------
    >>> image = np.ones((400, 400, 3), dtype=np.uint8)
    >>> polygon_points = np.array([[100, 100], [200, 50], [300, 150], [250, 300], [150, 250]])
    >>> result_image = DrawPolygon(image, polygon_points,  (255, 0, 0))
    """
    points = np.array([points])
    points = points.astype(np.int32)
    img = cv2.fillPoly(img, points, color)
    return img

def DrawCornerRectangle(img, bbox, length=30, lthickness=5, rthickness=1,
                        bboxcolor=Z.purple, cornercolor=Z.green):
    """
    Draws a rectangle with highlighted corners on an image. The rectangle is defined by a bounding box,
    and the corners are emphasized with lines of a specified length and thickness.

    Parameters:
    -----------
    img : numpy.ndarray
        The image (as a NumPy array) on which the rectangle and corners will be drawn.
    bbox : tuple
        The bounding box coordinates in the format `(x, y, x1, y1)`, where `(x, y)` is the top-left corner
        and `(x1, y1)` is the bottom-right corner.
    length : int, optional
        The length of the corner lines (default is 30 pixels).
    lthickness : int, optional
        The thickness of the corner lines (default is 5).
    rthickness : int, optional
        The thickness of the rectangle lines (default is 1).
    bboxcolor : tuple, optional
        The color of the rectangle in BGR format (e.g., (128, 0, 128) for purple). Default is `Z.purple`.
    cornercolor : tuple, optional
        The color of the corner lines in BGR format (e.g., (0, 255, 0) for green). Default is `Z.green`.

    Returns:
    --------
    numpy.ndarray
        The modified image with the rectangle and corners drawn.

    Usage Example:
    --------------
    >>> img = np.zeros((300, 500, 3), dtype=np.uint8)
    >>> bbox = (50, 50, 200, 150)
    >>> img = DrawCornerRectangle(img, bbox, length=30, lthickness=5, rthickness=1,
    ...                          bboxcolor=(128, 0, 128), cornercolor=(0, 255, 0))
    """
    x, y, x1, y1 = bbox
    cv2.rectangle(img, (x, y), (x1, y1), bboxcolor, rthickness)
    # Top-left corner (x, y)
    cv2.line(img, (x, y), (x + length, y), cornercolor, lthickness)
    cv2.line(img, (x, y), (x, y + length), cornercolor, lthickness)
    # Top-right corner (x1, y)
    cv2.line(img, (x1, y), (x1 - length, y), cornercolor, lthickness)
    cv2.line(img, (x1, y), (x1, y + length), cornercolor, lthickness)
    # Bottom-left corner (x, y1)
    cv2.line(img, (x, y1), (x + length, y1), cornercolor, lthickness)
    cv2.line(img, (x, y1), (x, y1 - length), cornercolor, lthickness)
    # Bottom-right corner (x1, y1)
    cv2.line(img, (x1, y1), (x1 - length, y1), cornercolor, lthickness)
    cv2.line(img, (x1, y1), (x1, y1 - length), cornercolor, lthickness)

    return img


if __name__=="__main__":
    imagePath = r"E:\PythonProject\pyzjrPyPi\pyzjr\test.png"
    image = np.ones((400, 400, 3), dtype=np.uint8)
    polygon_points = np.array([[100, 100], [200, 50], [300, 150], [250, 300], [150, 250]])
    result_image = DrawPolygon(image, polygon_points, (255, 0, 0))
    img = cv2.imread(imagePath)
    # img = np.zeros((300, 500, 3), dtype=np.uint8)
    bbox = (50, 50, 200, 150)
    img = DrawCornerRectangle(img, bbox, length=30, lthickness=5, rthickness=1,
                            bboxcolor=(128, 0, 128), cornercolor=(0, 255, 0))
    cv2.imshow("show", img)
    cv2.imshow("show2", result_image)
    cv2.waitKey(0)