"""
Copyright (c) 2023, Auorui.
All rights reserved.

Some processing of binarization
"""
import cv2
import numpy as np
from pyzjr.utils.check import is_bool, is_gray_image

def uint2single(image):
    # 255 -> (0, 1)
    return np.float32(image / 255)

def single2uint(image):
    # (0, 1) -> 255
    return np.uint8((image.clip(0, 1) * 255.).round())

def binarization(image, min_value=127, max_value=255):
    """
    Convert an input image to binary (black-and-white) using thresholding.
    The function first converts the image to grayscale, then applies binary thresholding
    where pixel values above min_value become white (max_value) and others become black.

    Args:
        image (numpy.ndarray): Input image in BGR format (OpenCV default).
        min_value (int, optional): Threshold value for binarization. Pixels above this
            value will be set to max_value. Defaults to 127.
        max_value (int, optional): Maximum pixel value for binary output (white color).
            Defaults to 255.

    Returns:
        numpy.ndarray: Binary image where pixels are either 0 (black) or max_value (white).
    """
    np_image = np.array(image).astype(np.uint8)
    if not is_gray_image(image):
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(np_image, min_value, max_value, cv2.THRESH_BINARY)
    return binary_image

def approximate(image, std=127.5, dtype=np.uint8):
    """
    Convert a single channel image into a binary image.
    """
    if not is_gray_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[image > std] = 255
    image[image < std] = 0
    image = image.astype(dtype)
    return image

def ceilfloor(image, dtype=np.uint8):
    """
    The pixel value of the input image is limited between the maximum value of 255 and the minimum value of 0
    """
    if not is_gray_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype(dtype)
    # == np.clip(image, 0, 255).astype(dtype)
    return image

def up_low(image, lower, upper, dtype=np.uint8):
    """Create a binary mask for pixels within the specified BGR color range.

    Args:
        image: Input image (BGR or grayscale format)
        lower: Lower bound of color range as (B, G, R) or single value for grayscale
        upper: Upper bound of color range as (B, G, R) or single value for grayscale
        dtype: Data type for threshold comparison. Defaults to np.uint8.

    Returns:
        Binary mask where white pixels (255) are within range, black (0) otherwise
    """
    np_image = np.array(image).astype(dtype)
    lower = np.array(lower, dtype=dtype)
    upper = np.array(upper, dtype=dtype)
    _mask = cv2.inRange(np_image, lower, upper)
    return _mask

def remove_mask_parti_color(image, lower, upper, replacement_color=(255, 255, 255)):
    """Remove particles of specific color range from an image and replace them with given color.

    This function identifies pixels within the specified BGR color range and replaces them
    with the replacement color, typically used for cleaning labels or removing artifacts.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        lower: Lower bound of BGR color range as a tuple (B, G, R). Values 0-255.
        upper: Upper bound of BGR color range as a tuple (B, G, R). Values 0-255.
        replacement_color: Color to replace matched pixels (B, G, R). Defaults to white (255, 255, 255).

    Returns:
        Processed image with specified color range replaced.
    """
    if not all(0 <= x <= 255 for x in (*lower, *upper, *replacement_color)):
        raise ValueError("All color values must be between 0-255")
    mask = up_low(image, lower, upper)
    image[mask > 0] = replacement_color
    return image

def move_mask_foreground(image, H_pixels=None, V_pixels=None):
    """
    Moves the foreground (non-zero pixels) of an image by specified horizontal and vertical offsets,
    while maintaining the original image dimensions. Areas outside the image bounds are filled with black.
    Args:
        image (numpy.ndarray): Input image as a NumPy array (grayscale or color).
        H_pixels (int, optional): Horizontal shift amount. Positive values shift right,
            negative values shift left. Defaults to 0.
        V_pixels (int, optional): Vertical shift amount. Positive values shift down,
            negative values shift up. Defaults to 0.
    """
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

def adaptive_bgr_threshold(image, bgr_threshold, total_threshold=None):
    """Apply adaptive thresholding based on individual BGR channel thresholds and optional total intensity.

    Creates a binary mask where pixels must meet all specified conditions:
    1. Each BGR component exceeds its respective threshold
    2. (Optional) The sum of BGR components exceeds total_threshold

    Adjusting parameters can solve the problem of difficult boundary differentiation in sub pixels

    Args:
        image (numpy.ndarray): Input BGR image (3-channel, uint8)
        bgr_threshold (tuple): Threshold values for (Blue, Green, Red) channels (0-255 each)
        total_threshold (int, optional): Minimum sum of BGR values. If None, uses sum of bgr_threshold.
                                      Defaults to None.

    Returns:
        numpy.ndarray: Binary mask where white (255) pixels meet all thresholds, black (0) otherwise
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be 3-channel BGR image. Shape: {}".format(image.shape))
    if len(bgr_threshold) != 3:
        raise TypeError("bgr_threshold must contain exactly 3 values (B,G,R)")
    if not all(0 <= t <= 255 for t in bgr_threshold):
        raise ValueError("All BGR thresholds must be 0-255")
    if total_threshold is not None and not 0 <= total_threshold <= 765:  # 255*3
        raise ValueError("total_threshold must be 0-765")
    b_thresh, g_thresh, r_thresh = bgr_threshold
    total_threshold = sum(bgr_threshold) if total_threshold is None else total_threshold

    sum_mask = image.sum(axis=2) >= total_threshold
    b_mask = image[:, :, 0] >= b_thresh  # B
    g_mask = image[:, :, 1] >= g_thresh  # G
    r_mask = image[:, :, 2] >= r_thresh  # R

    combined_mask = sum_mask & b_mask & g_mask & r_mask
    return np.where(combined_mask, 255, 0).astype(np.uint8)

def bool2mask(matrix, value=255):
    """Convert a boolean matrix to an 8-bit unsigned integer mask image.

    Transforms a boolean array (True/False) into a grayscale mask image where:
    - True values are converted to the specified value (default: 255, white)
    - False values remain 0 (black)

    Args:
        matrix (numpy.ndarray): Input boolean array (dtype=bool)
        value (int, optional): Intensity value for True pixels. Must be between 0-255.
                            Defaults to 255 (white).

    Returns:
        numpy.ndarray: 8-bit unsigned integer mask (dtype=np.uint8)
    """
    if not is_bool(matrix):
        raise ValueError("Input matrix must be of bool dtype. Got: {}".format(matrix.dtype))
    if not 0 <= value <= 255:
        raise ValueError("Value must be between 0-255. Got: {}".format(value))
    result_int = matrix.astype(int)
    _mask = result_int * value
    return _mask.astype(np.uint8)


def create_rectmask(image, StartPoint=None, EndPoint=None, bboxes=None):
    """Create a rectangular mask and apply it to the input image.

    Generates a binary mask covering specified rectangular region(s) and returns both
    the mask and the masked image. Supports either individual points or bounding box
    coordinates for rectangle definition.

    Args:
        image (numpy.ndarray or PIL.Image): Input image (color or grayscale).
        StartPoint (list/tuple, optional): Top-left point [x1, y1]. Required if bboxes is None.
        EndPoint (list/tuple, optional): Bottom-right point [x2, y2]. Required if bboxes is None.
        bboxes (list/tuple, optional): Combined coordinates [x1, y1, x2, y2].
                                      Takes precedence over StartPoint/EndPoint.

    Returns:
        tuple: (mask, masked_image) where:
            - mask (numpy.ndarray): Binary mask (0=background, 255=rectangle)
            - masked_image (numpy.ndarray): Original image with mask applied
    """
    image = np.array(image).astype(np.uint8)
    h, w = image.shape[:2]
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    if bboxes is not None:
        if len(bboxes) != 4:
            raise ValueError("bboxes must contain exactly 4 coordinates [x1,y1,x2,y2]")
        x1, y1, x2, y2 = bboxes
    elif StartPoint is not None and EndPoint is not None:
        if len(StartPoint) != 2 or len(EndPoint) != 2:
            raise ValueError("StartPoint and EndPoint must contain exactly 2 coordinates each")
        x1, y1 = StartPoint
        x2, y2 = EndPoint
    else:
        raise ValueError("Must provide either bboxes or both StartPoint and EndPoint")
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    mask[y1: y2, x1: x2] = 255
    if mask.ndim == 3 and mask.shape[-1] > 1:
        mask = mask[:, :, 0]
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return mask, masked_image

def auto_canny(image, sigma=0.33):
    """
    Automatically detect edges in the input image using the Canny edge detection algorithm.

    The lower and upper thresholds for Canny edge detection are determined based on the median
    pixel intensity of the image and a specified sigma value.

    Args:
        image (numpy.ndarray): Input image as a NumPy array (grayscale).
        sigma (float, optional): A scaling factor to determine the lower and upper thresholds.
            Default is 0.33.

    Returns:
        numpy.ndarray: A binary image with edges detected.
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def inpaint_defect(image, mask, radius=10, flags=cv2.INPAINT_TELEA):
    """Perform image inpainting to reconstruct missing or damaged regions.

    Uses OpenCV's inpainting algorithms to fill in selected regions of an image
    based on the surrounding content. Useful for removing defects, watermarks,
    or unwanted objects.

    Args:
        image (numpy.ndarray): Input defect image (BGR or grayscale format).
        mask (numpy.ndarray): Binary mask specifying areas to inpaint (white=defect, black=background).
                             Must be 8-bit single-channel and same size as image.
        radius (int, optional): Radius of circular neighborhood to consider for inpainting.
                               Defaults to 10 pixels.
        flags (int, optional): Inpainting algorithm to use. Either:
                              - cv2.INPAINT_TELEA (1): Alexandru Telea's method (faster)
                              - cv2.INPAINT_NS (0): Navier-Stokes based method (smoother)
                              Defaults to cv2.INPAINT_TELEA.

    Returns:
        numpy.ndarray: The inpainted image with same shape and type as input.
    """
    dst = cv2.inpaint(image, mask, radius, flags)
    return dst


if __name__=="__main__":
    image = cv2.imread(r"E:\PythonProject\pyzjrPyPi\pyzjr\utils\tryout\images\shapes.png")
    mask, masked_image = create_rectmask(image, (50, 20), (180, 150))
    ulimg = up_low(image, (0, 0, 0), (255, 255, 123))
    approximg = approximate(image)
    ceilimg = ceilfloor(image)
    print(ceilimg.shape)
    # cv2.imshow("ss", mask)
    cv2.imshow("mask", ulimg)
    cv2.waitKey(0)













