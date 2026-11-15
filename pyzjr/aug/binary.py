"""
Copyright (c) 2023, Auorui.
All rights reserved.

Some processing of binarization
"""
import cv2
import numpy as np


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

    is_gray = (np_image.ndim == 2) or (np_image.ndim == 3 and np_image.shape[-1] == 1)

    if not is_gray:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(np_image, min_value, max_value, cv2.THRESH_BINARY)
    return binary_image


def clip_pixels(image, min_val=0, max_val=255, dtype=np.uint8, to_gray=False):
    """
    Clip pixel values to a specified range. This function combines the functionality
    of `approximate` and `ceilfloor`.

    Args:
        image (np.ndarray): Input image.
        min_val (int/float): Minimum allowed pixel value.
        max_val (int/float): Maximum allowed pixel value.
        dtype: Output data type.
        to_gray (bool): Whether to convert to grayscale first.

    Returns:
        np.ndarray: Image with clipped pixel values.
    """
    img = np.array(image)
    if to_gray and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.clip(img, min_val, max_val)
    return img.astype(dtype)


def in_range_mask(image, lower, upper):
    """
    Create a binary mask for pixels within the specified color or intensity range.

    Args:
        image (np.ndarray): Input image (BGR or grayscale).
        lower (tuple/int): Lower bound.
        upper (tuple/int): Upper bound.

    Returns:
        np.ndarray: Binary mask (255 for in-range, 0 for out-of-range).
    """
    if len(image.shape) == 2 or image.shape[2] == 1:
        if isinstance(lower, (tuple, list)): lower = lower[0]
        if isinstance(upper, (tuple, list)): upper = upper[0]

    return cv2.inRange(image, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))


def replace_color_range(image, lower, upper, replacement_color=(255, 255, 255)):
    """
    Replace pixels within a specific color range with a new color.
    Commonly used for cleaning labels or removing artifacts.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        lower (tuple): Lower bound of BGR color range as (B, G, R). Values 0-255.
        upper (tuple): Upper bound of BGR color range as (B, G, R). Values 0-255.
        replacement_color (tuple): Replacement color (B, G, R). Defaults to white.

    Returns:
        numpy.ndarray: Processed image with specified color range replaced.
    """
    result = image.copy()
    mask = in_range_mask(result, lower, upper)
    result[mask > 0] = replacement_color
    return result


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
        raise ValueError(f"Input must be 3-channel BGR image. Shape: {image.shape}")
    if len(bgr_threshold) != 3:
        raise TypeError("bgr_threshold must contain exactly 3 values (B,G,R)")
    if not all(0 <= t <= 255 for t in bgr_threshold):
        raise ValueError("All BGR thresholds must be 0-255")

    b_thresh, g_thresh, r_thresh = bgr_threshold
    total_threshold = sum(bgr_threshold) if total_threshold is None else total_threshold

    if not 0 <= total_threshold <= 765:  # 255*3
        raise ValueError("total_threshold must be 0-765")

    sum_mask = image.sum(axis=2, dtype=np.int32) >= total_threshold

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
    if matrix.dtype != np.bool_:
        raise ValueError("Input matrix must be of bool dtype. Got: {}".format(matrix.dtype))
    if not 0 <= value <= 255:
        raise ValueError("Value must be between 0-255. Got: {}".format(value))
    return np.where(matrix, value, 0).astype(np.uint8)


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


def inpaint_defect(image, mask, radius=10, flags=1):
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