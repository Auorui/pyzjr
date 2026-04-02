"""
Copyright (c) 2023, Auorui.
All rights reserved.
"""
import cv2
from pylab import *
from skimage.morphology import disk
from skimage.filters import rank
from skimage import measure
from pyzjr.utils.check import is_gray_image


def incircleV1(image, contours_arr, is_draw=True, color=(0, 0, 255), use_check_point=False):
    """
    Finds and optionally draws the maximum inscribed circle for each contour.
    The method calculates the largest circle that fits inside each contour and returns its diameter and position.

    Args:
        image: Single-channel image, typically a binary image, used for calculating the maximum inscribed circle.
        contours_arr: List of contours, usually obtained from `cv2.findContours`. Each contour is a 2D point set.
        is_draw: Whether to draw the inscribed circles on the result image. Defaults to True.
                If True, draws the inscribed circles on the image.
        color: BGR color tuple for drawing circles. Defaults to red (0, 0, 255).
        use_check_point: Whether to perform inscribed circle validation by checking if the circle center and all points
                        within the radius are inside the contour. Mainly used for mesh-like crack structures.
                        Defaults to False.

    Returns:
        If `is_draw=True`, returns the image with drawn circles and the circle diameters.
        If `is_draw=False`, returns only the circle diameters.
    """
    if is_gray_image(image):
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result = image.copy()

    max_val = 0
    max_dist_pt = None
    # Consider only non-zero points.
    non_zero_points = np.array(np.nonzero(image))
    non_zero_points = non_zero_points.T
    for contours in contours_arr:
        raw_dist = np.zeros(image.shape[:2], dtype=np.float32)
        # Use cv2.pointPolygonTest to calculate the distance from each non-zero point to the contour.
        for point in non_zero_points:
            x, y = point
            x, y = int(x), int(y)
            raw_dist[x, y] = cv2.pointPolygonTest(contours, (y, x), True)
        # for i in range(image.shape[0]):
        #     for j in range(image.shape[1]):
        #         raw_dist[i, j] = cv2.pointPolygonTest(contours, (j, i), True)
        # Find the diameter of the largest inscribed circle and its center.
        min_val, curr_max_val, _, curr_max_dist_pt = cv2.minMaxLoc(raw_dist)
        if curr_max_val > max_val:
            # Check if the eight points obtained by adding the radius to the center of the circle are all within the contour.
            if use_check_point:
                if check_points_in_contour(curr_max_dist_pt, int(curr_max_val)-2, contours):
                    max_val = curr_max_val
                    max_dist_pt = curr_max_dist_pt
            else:
                max_val = curr_max_val
                max_dist_pt = curr_max_dist_pt
    wide = max_val * 2
    if is_draw and max_dist_pt is not None:
        result = cv2.circle(result, max_dist_pt, int(max_val), color, 1, 1, 0)

    if is_draw:
        return wide, result
    else:
        return wide

def incircleV2(image, is_draw=True, color=(0, 0, 255)):
    """
    Finds and optionally draws the maximum inscribed circle in a binary image.

    The function uses distance transform to locate the largest circle that fits entirely
    within the foreground (white) region of the binary image.

    Args:
        binary_image: Single-channel binary image (0=background, 255=foreground).
                     Should be uint8 dtype.
        draw: Whether to draw the circle on the image. Defaults to True.
        color: BGR color tuple for the circle. Defaults to red (0, 0, 255).

    Returns:
        If draw=True:
            Tuple containing:
            - Image with drawn circle (3-channel BGR)
            - Tuple of ((center_x, center_y), radius)
        If draw=False:
            Tuple of ((center_x, center_y), radius)
    """
    if is_gray_image(image):
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result = image.copy()
    max_val = 0
    max_dist_pt = None
    dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 3)
    min_val, curr_max_val, _, curr_max_dist_pt = cv2.minMaxLoc(dist_transform)
    if curr_max_val > max_val:
        max_val = curr_max_val
        max_dist_pt = curr_max_dist_pt
    wide = max_val * 2
    if is_draw and max_dist_pt is not None:
        result = cv2.circle(result, max_dist_pt, int(max_val), color, 1, 1, 0)
    if is_draw:
        return wide, result
    else:
        return wide

def outcircle(img, contours_arr, is_draw=True, color=(0, 255, 0)):
    """
    Finds and optionally draws minimum enclosing circles for each contour.

    Args:
        image: Input image (will be converted to 3-channel if grayscale).
        contours: List of contours where each contour is a numpy array of shape (N,1,2).
        draw: Whether to draw circles on the image. Defaults to True.
        color: BGR color tuple for drawing circles. Defaults to green (0, 255, 0).

    Returns:
        If draw=True:
            Tuple containing:
            - Image with drawn circles (3-channel BGR)
            - List of circle parameters as ((center_x, center_y), radius)
        If draw=False:
            List of circle parameters only
    """
    radii = []
    if is_draw:
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        result = None

    for cnt in contours_arr:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        radii.append([(x, y), radius])
        if is_draw:
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(result, center, radius, color, 1)  # Draw an excircle
            cv2.circle(result, center, 1, color, 1)       # Draw the center of the circle

    if is_draw:
        return radii, result
    else:
        return radii

def check_points_in_contour(center, radius, contour):
    x, y = center
    directions = [
        (x + radius, y),  # Right
        (x - radius, y),  # Left
        (x, y + radius),  # Bottom
        (x, y - radius),  # Upper
        (x + radius, y + radius),  # bottom right
        (x + radius, y - radius),  # upper right
        (x - radius, y + radius),  # left lower
        (x - radius, y - radius),  # upper left
    ]
    for pt in directions:
        if cv2.pointPolygonTest(contour, pt, False) < 0:  # The point is outside the outline.
            return False
    return True


def calculate_contour_lengths(contours, min_area: float = 5.0):
    """Calculate lengths of contours

    Args:
        contours: List of contour points where each contour is a numpy array
                 with shape (N, 1, 2) representing (x,y) coordinates.
        min_area: Minimum area threshold (contours with area <= min_area will be
                ignored). Defaults to 5.0.
    """
    if not isinstance(contours, list):
        raise TypeError("contours must be a list of numpy arrays")
    if min_area < 0:
        raise ValueError("min_area must be non-negative")
    contour_lengths = []
    for contour in contours:
        if not isinstance(contour, np.ndarray):
            raise TypeError(f"Each contour must be numpy array, got {type(contour)}")
        # Calculate contour area and length
        area = cv2.contourArea(contour)
        if area > min_area:
            length = cv2.arcLength(contour, closed=True)
            contour_lengths.append(length)
    # total_length = float(np.sum(contour_lengths))
    return contour_lengths

def label_contours(image, contours, color=(0, 255, 0), thickness=2):
    """Draw labeled contours on the image with centroid markers.

    Args:
        image: Input image (will be converted to BGR if grayscale)
        contours: List of contours where each contour has shape (N,1,2)

    Returns:
        Image with drawn contours and centroid labels

    Raises:
        TypeError: If contours is not a list of numpy arrays
        cv2.error: If any contour is invalid
    """
    # Convert to BGR if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Process each contour individually
    for i, cnt in enumerate(contours):
        if not isinstance(cnt, np.ndarray):
            raise TypeError(f"Contour {i} is not a numpy array")
        # Calculate moments for the current contour
        M = cv2.moments(cnt)
        if M["m00"] == 0:  # Skip zero-area contours
            continue
        # Draw contour
        cv2.drawContours(image, [cnt], -1, color, thickness)
        # Calculate and mark centroid
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(image, (cX, cY), 4, color, -1)
        cv2.putText(image, str(i), (cX - 10, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
    return image

def sort_contours(contours, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
    return contours, boundingBoxes


def drawOutline(blackbackground, contours, color=(255, 0, 255), thickness=1):
    """Draw contour"""
    cv2.drawContours(blackbackground, contours, -1, color, thickness=thickness)

def SearchOutline(binary, level=128):
    """
    Detects and extracts contour coordinates from a binary image using marching squares.

    Args:
        binary_image: Input binary image (0=background, non-zero=foreground).
                     Should be 2D numpy array.
        level: Value along which to find contours. Defaults to 128.
        fully_connected: Either 'low' or 'high' to determine contour connectivity.
                       'low' treats diagonally adjacent pixels as connected.
                       'high' requires orthogonal connection.
        positive_orientation: Either 'low' or 'high' to determine contour orientation.

    Returns:
        List of contour arrays where each contour is an Nx2 array of (x,y) coordinates.
    """
    contours = measure.find_contours(binary, level=level, fully_connected='low', positive_orientation='low')
    contours_xy = [np.fliplr(np.vstack(contour)).astype(np.int32) for contour in contours]
    return contours_xy


def gradient_outline(binary, radius=2):
    """
    Detects gradient-based outlines from a binary image using morphological operations.

    The function performs two main steps:
    1. Noise reduction using median filtering
    2. Gradient magnitude calculation using local rank filter

    Args:
        binary_img: Input binary image (0=background, 255=foreground).
                   Should be uint8 single-channel.
        radius: Radius of the disk-shaped structuring element (default=2).
               Larger values produce thicker outlines but may lose detail.

    Returns:
        Outline intensity image where higher values indicate stronger edges.
    """
    if binary.dtype != np.uint8 or len(binary.shape) != 2:
        raise ValueError("Input must be single-channel uint8 binary image")
    if not isinstance(radius, int):
        raise TypeError("Radius must be integer")
    denoised = rank.median(binary, disk(radius))
    gradient = rank.gradient(denoised, disk(radius))
    gradient = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    return gradient

def find_contours_custom(binary_img: np.ndarray,
                         mode: int = cv2.RETR_EXTERNAL,
                         method: int = cv2.CHAIN_APPROX_SIMPLE):
    """Customizable contour detection with flexible retrieval modes and approximation methods.

    Provides a wrapper around OpenCV's findContours with configurable parameters for
    different use cases. Returns only the contours list (without hierarchy for simplicity).

    Args:
        binary_img: Input binary image (0=background, non-zero=foreground).
                   Should be uint8 single-channel.
        mode: Contour retrieval mode (default: RETR_EXTERNAL). Options:
              - cv2.RETR_EXTERNAL: Retrieves only extreme outer contours
              - cv2.RETR_TREE: Retrieves all contours and reconstructs full hierarchy
              - cv2.RETR_LIST: Retrieves all contours without hierarchy
        method: Contour approximation method (default: CHAIN_APPROX_SIMPLE). Options:
              - cv2.CHAIN_APPROX_NONE: Stores absolutely all contour points
              - cv2.CHAIN_APPROX_SIMPLE: Compresses horizontal, vertical, and diagonal segments
                                         leaving only their end points

    Returns:
        List of contours where each contour is a numpy array of shape (N,1,2)
        containing (x,y) coordinates of the contour points.

        contours_arr, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  #  Each contour and pixel point
        contours_arr, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Outermost contour, endpoints
    """
    if binary_img.dtype != np.uint8 or len(binary_img.shape) != 2:
        raise ValueError("Input must be single-channel uint8 binary image")

    contours, _ = cv2.findContours(binary_img, mode, method)
    return contours



if __name__ == "__main__":
    from pyzjr.augmentation.utils.binary import binarization
    image = cv2.imread(r"E:\PythonProject\pyzjrPyPi\pyzjr\utils\tryout\images\shapes.png")
    thresh = binarization(image)
    # contours_arr = SearchOutline(thresh)
    # contours_arr, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    gradient_thresh = gradient_outline(thresh)
    # wide, result = incircleV1(thresh, contours_arr)
    # print(wide)
    # wide, result_transform = incircleV2(thresh)
    # print(wide)
    # radii, results = outcircle(thresh, contours_arr)
    # print(wide, radii)
    # cv2.imwrite("sss.png", result_transform)
    # image = label_contours(image, thresh)
    cv2.imshow("ss", gradient_thresh)
    cv2.waitKey(0)