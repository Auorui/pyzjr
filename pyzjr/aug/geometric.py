"""
Copyright (c) 2023, Auorui.
All rights reserved.

Geometric transformations based on OpenCV (rotation, scaling, cropping, resizing, filling, erasing, etc.)
"""
import cv2
import random
import warnings
import numpy as np
from random import uniform
from typing import List
from pyzjr.data.helper import to_2tuple
from pyzjr.utils.decorators import _sync_wrapper


def _parse_label_flags(imglists, is_labels):
    if is_labels is None:
        raise ValueError(
            "The `is_labels` parameter is required. "
            "Please pass a boolean (e.g., True) or a list of booleans (e.g., [False, True])."
        )
    if isinstance(is_labels, bool):
        return [False] * (len(imglists) - 1) + [is_labels] if imglists else []
    elif isinstance(is_labels, (list, tuple)):
        if len(is_labels) != len(imglists):
            raise ValueError(f"Length of is_labels ({len(is_labels)}) must match length of imglists ({len(imglists)}).")
        return is_labels
    else:
        raise TypeError(
            f"Invalid type for `is_labels`: {type(is_labels)}. "
            "Must be bool or a list/tuple of bool."
        )

# ----------------------------------Resize----------------------------------
def smart_resize(image, target_shape, is_label=False):
    """
    Smart resizing for images or labels.

    Args:
        image (np.ndarray): Input image as a NumPy array.
        target_shape (tuple): Target size as (target_w, target_h).
        is_label (bool): Whether the input is a label map. If True, forces nearest-neighbor interpolation.

    Returns:
        np.ndarray: The resized image or label.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("np_img should be a NumPy array. Got {}.".format(type(image)))
    h, w = image.shape[:2]
    target_w, target_h = to_2tuple(target_shape)
    if (h, w) == (target_h, target_w):
        return image

    # For labels, force nearest-neighbor to prevent creating new class pixels
    if is_label:
        interpolation = cv2.INTER_NEAREST
    else:
        if target_h * target_w > h * w:
            # Upscale: use bicubic for smoother results
            interpolation = cv2.INTER_CUBIC
        else:
            # Downscale: use area interpolation to avoid moiré and aliasing
            interpolation = cv2.INTER_AREA

    return cv2.resize(image, (int(target_w), int(target_h)), interpolation=interpolation)


def resize_samescale(image, width=None, height=None, is_label=False):
    """
    Resize the input image while maintaining the aspect ratio.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        width (int, optional): Target width of the resized image. If None, it will be calculated based on the height.
        height (int, optional): Target height of the resized image. If None, it will be calculated based on the width.
        is_label (bool): Whether the input is a label map. If True, forces nearest-neighbor interpolation.

    Returns:
        numpy.ndarray: Resized image with the same aspect ratio as the input image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("np_img should be a NumPy array. Got {}.".format(type(image)))
    h, w = image.shape[:2]

    if width is None and height is None:
        return image
    elif width is not None and height is not None:
        warnings.warn("Both width and height are specified. The image will be resized based on the width while maintaining the aspect ratio. "
                      "The specified height may not be exactly met.")
    dim = None
    if width is not None:
        r = width / float(w)
        dim = (width, int(round(h * r)))
    elif height is not None:
        r = height / float(h)
        dim = (int(round(w * r)), height)
    return smart_resize(image, dim, is_label)


def resize_pad(image, target_shape, pad_color=(128, 128, 128), is_label=False):
    """
    Resize the image while maintaining aspect ratio, then pad it to the target shape.
    This is commonly known as 'letterbox' padding.

    Args:
        image (np.ndarray): Input image or label array.
        target_shape (tuple): Target size as (target_w, target_h).
        pad_color (tuple): BGR color for padding. Default is (128, 128, 128).
        is_label (bool): Whether the input is a label map. If True, forces nearest-neighbor
                         interpolation and pads with black (0, 0, 0).

    Returns:
        tuple: (padded_image, (top, left, new_h, new_w, orig_h, orig_w))
            - padded_image: The resized and padded image.
            - (top, left, new_h, new_w, orig_h, orig_w): Metadata for cropping later during inference.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("np_img should be a NumPy array. Got {}.".format(type(image)))
    orig_h, orig_w = image.shape[:2]
    target_w, target_h = to_2tuple(target_shape)
    # Compare aspect ratios: w / h > t_w / t_h is equivalent to w * t_h > t_w * h
    if orig_w * target_h > target_w * orig_h:
        image = resize_samescale(image, width=target_w, height=None, is_label=is_label)
    else:
        image = resize_samescale(image, width=None, height=target_h, is_label=is_label)

    new_h, new_w = image.shape[:2]
    is_normalized = image.dtype == np.float32 and np.max(image) <= 1.0

    if is_normalized:
        pad_color = tuple(c / 255.0 for c in pad_color)
        output_dtype = np.float32
    else:
        output_dtype = np.uint8

    if is_label:
        pad_color = (0, 0, 0)
        output_dtype = np.uint8

    if len(image.shape) == 3:
        background_image = np.full((target_h, target_w, 3), pad_color, dtype=output_dtype)
    else:
        single_color = pad_color[0] if isinstance(pad_color, tuple) else pad_color
        background_image = np.full((target_h, target_w), single_color, dtype=output_dtype)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2

    background_image[top:top + new_h, left:left + new_w] = image
    return background_image, (top, left, new_h, new_w, orig_h, orig_w)


def resize_unpad(padded_image, pad_meta, is_label=False):
    """
    Reverse the padding and resize operations applied by `resize_pad`.

    Args:
        padded_image (np.ndarray): The predicted or padded image array (H, W) or (H, W, C).
        pad_meta (tuple): The metadata returned by `resize_pad` as (top, left, new_h, new_w, orig_h, orig_w).
        is_label (bool): If True, uses nearest-neighbor interpolation to prevent
                         introducing new class labels.

    Returns:
        np.ndarray: The unpadded and resized image, matching the original shape.
    """
    pad_top, pad_left, new_h, new_w, orig_h, orig_w = pad_meta
    cropped = padded_image[pad_top:pad_top + new_h, pad_left:pad_left + new_w]

    return smart_resize(cropped, (orig_w, orig_h), is_label=is_label)


@_sync_wrapper
def resize_pad_lists(imglists, target_shape, pad_color=(128, 128, 128), is_labels=None,
                     return_meta=False):
    """
    Batch version of `resize_pad` for a list of images/labels.

    Args:
        imglists (np.ndarray or list): Input image(s).
        target_shape (tuple): Target size as (target_w, target_h).
        pad_color (tuple): BGR color for padding images. Labels always pad with 0.
        is_labels (bool or list, optional): Whether each input is a label map.
            If a bool is given, it applies to the last element (consistent with
            `_parse_label_flags`). Defaults to None.
        return_meta (bool): If True, also returns a list of pad metadata
            (one per image) for later `resize_unpad`.

    Returns:
        list: Padded image(s).
        list (optional): pad_meta list when return_meta=True.
    """
    is_labels = _parse_label_flags(imglists, is_labels)
    results, metas = [], []
    for i, img in enumerate(imglists):
        padded, meta = resize_pad(img, target_shape, pad_color=pad_color,
                                  is_label=is_labels[i])
        results.append(padded)
        metas.append(meta)
    if return_meta:
        return results, metas
    return results


@_sync_wrapper
def centerzoom(imglists, zoom_factor: float, is_labels=None):
    """
    Center Zoom Image.
    Crops a central region and resizes it back to the original size.

    Args:
        imglists (np.ndarray or list): Input image(s) or a list of images/labels.
        zoom_factor (float): Scale factor.
            > 1 means zoom in (e.g., 2.0 shows a 1/2 center area).
            < 1 means zoom out (e.g., 0.5 shrinks the image content).
        is_labels (bool or list, optional): Whether the input is a label map.
            If a list is provided, it must match the length of `imglists`. Defaults to None.

    Returns:
        The image scaled by the center.
    """
    is_labels = _parse_label_flags(imglists, is_labels)
    h, w = imglists[0].shape[:2]
    crop_h = max(1, int(round(h / zoom_factor)))
    crop_w = max(1, int(round(w / zoom_factor)))
    y_start = (h - crop_h) // 2
    x_start = (w - crop_w) // 2

    zoomed_list = []
    for i, img in enumerate(imglists):
        cropped = crop_by_1point(img, (x_start, y_start), (crop_w, crop_h))
        zoomed_list.append(smart_resize(cropped, (w, h), is_label=is_labels[i]))
    return zoomed_list

@_sync_wrapper
def ensure_min_size(imglists, target_shape, is_labels=None):
    """
    Helper method to ensure the image is large enough to be cropped.
    Resizes proportionally if any dimension is smaller than target_shape.

    Args:
        imglists (np.ndarray or list): Input image(s).
        target_shape (tuple): Target size as (target_w, target_h).
        is_labels (bool or list, optional): Whether the input is a label map. Defaults to None.

    Returns:
        np.ndarray or list: The resized image(s).
    """
    is_labels = _parse_label_flags(imglists, is_labels)
    H, W = imglists[0].shape[:2]
    target_w, target_h = to_2tuple(target_shape)

    if W < target_w or H < target_h:
        scale_w = target_w / W
        scale_h = target_h / H
        scale = max(scale_w, scale_h)
        new_w = int(np.ceil(W * scale))
        new_h = int(np.ceil(H * scale))
        for i, img in enumerate(imglists):
            imglists[i] = smart_resize(img, (new_w, new_h), is_label=is_labels[i])
    return imglists

# ----------------------------------Flip & Rotate----------------------------------
def flip(image, axis):
    """
    Flip the image
    Args:
        image: numpy array of image
        axis: random integer between 0 to 2
            vertical                          0
            horizontal                        1
            diagonal                          2
    Returns: numpy array of flipped image
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("np_img should be a NumPy array. Got {}.".format(type(image)))
    if axis == 0:
        image = np.flip(image, axis)
    elif axis == 1:
        image = np.flip(image, axis)
    elif axis == 2:
        image = np.flip(image, (0, 1))
    else:
        image = image
    return image.copy()


def vertical_flip(image):
    """Flip the image vertically"""
    return flip(image, 0)


def horizontal_flip(image):
    """Flip the image horizontally"""
    return flip(image, 1)


def diagonal_flip(image):
    """Flip the image vertically"""
    return flip(image, 2)

@_sync_wrapper
def random_flip(imglists, axis: int = None, prob: float = 0.5) -> List[np.ndarray]:
    """
    Randomly flip the image with a given probability.
    If prob <= 0, returns the original image.
    """
    if prob > 0 and random.random() < prob:
        if axis is None:
            axis = random.randint(0, 2)
        return [flip(img, axis) for img in imglists]
    return [img.copy() for img in imglists]

def rotate(image, angle, pad_color=(0, 0, 0), keep_bound=False):
    """
    Rotate the input image by angle degrees.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        angle (float): Rotation angle in degrees, counter-clockwise.
        pad_color (tuple, optional): Fill color for areas outside the rotated image.
                                     Default is (0, 0, 0) for black.
        keep_bound (bool, optional): If True, expands the output canvas to fit the
                                     entire rotated image without cropping. Default is False.

    Returns:
        numpy.ndarray: Rotated image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("np_img should be a NumPy array. Got {}.".format(type(image)))
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

    if keep_bound:
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int(np.ceil((h * sin) + (w * cos)))
        nH = int(np.ceil((h * cos) + (w * sin)))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        rotated_image = cv2.warpAffine(image, M, (nW, nH), borderValue=pad_color)
    else:
        rotated_image = cv2.warpAffine(image, M, (w, h), borderValue=pad_color)

    return rotated_image


@_sync_wrapper
def random_rotate(imglists, degrees=None, pad_color=(0, 0, 0), keep_bound=False, is_labels=None):
    """
    Randomly rotate the image(s) based on whether it is a square and the provided degree range.

    Args:
        imglists (np.ndarray or list): Input image(s).
        degrees (tuple, optional): The range of rotation angles (min_degree, max_degree).
            If None, selects a default angle based on the image shape. Defaults to None.
        pad_color (tuple, optional): Fill color for areas outside the rotated image.
            Defaults to (0, 0, 0) for black.
        keep_bound (bool, optional): If True, expands the output canvas to fit the
            entire rotated image without cropping. Defaults to False.
        is_labels (bool or list, optional): Whether the input is a label map.
            Labels use nearest-neighbor interpolation and black padding. Defaults to None.

    Returns:
        list: The rotated image(s) in BGR format.
    """
    is_labels = _parse_label_flags(imglists, is_labels)
    h, w = imglists[0].shape[:2]
    is_square = True if h == w else False
    if degrees is not None:
        angle = random.uniform(degrees[0], degrees[1])
    else:
        if is_square:
            angle = random.choice([0, 90, 180, 270])
        else:
            angle = random.choice([0, 180])

    result = []
    for i, img in enumerate(imglists):
        if is_labels[i]:
            # 标签使用最近邻插值
            img_h, img_w = img.shape[:2]
            M = cv2.getRotationMatrix2D((img_w/2, img_h/2), -angle, 1.0)
            if keep_bound:
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                nW = int(np.ceil((img_h * sin) + (img_w * cos)))
                nH = int(np.ceil((img_h * cos) + (img_w * sin)))
                M[0, 2] += (nW / 2) - (img_w / 2)
                M[1, 2] += (nH / 2) - (img_h / 2)
                res_img = cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_NEAREST, borderValue=0)
            else:
                res_img = cv2.warpAffine(img, M, (img_w, img_h), flags=cv2.INTER_NEAREST, borderValue=0)
            result.append(res_img)
        else:
            result.append(rotate(img, angle, pad_color, keep_bound))
    return result


@_sync_wrapper
def random_rot90(imglists: List[np.ndarray]) -> List[np.ndarray]:
    """Randomly rotate the input image by 90 degrees"""
    r = random.randint(0, 3)
    return [np.rot90(img, r, (0, 1)).copy() for img in imglists]

# ----------------------------------Crop----------------------------------
def crop_by_2point(image, point1, point2):
    """
    Crop the image based on the starting and ending points
    Args:
        image: The image to be cropped
        point1: Top left corner coordinates (x, y) of the cropping area
        point2: The coordinates of the bottom right corner of the cropping area (x1, y1)

    Returns:Cropped image
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("np_img should be a NumPy array. Got {}.".format(type(image)))
    height, width = image.shape[:2]
    x_min, y_min = point1
    x_max, y_max = point2

    assert x_max > x_min, "Maximum value of cropping x_max cannot be less than the minimum value x_min"
    assert y_max > y_min, "Maximum value of cropping y_max cannot be less than the minimum value y_min"
    assert x_min >= 0, "x_min cannot be less than 0"
    assert x_max <= width, "x_max cannot be greater than the image width"
    assert y_min >= 0, "y_min cannot be less than 0"
    assert y_max <= height, "y_max cannot be greater than the image height"

    return image[y_min:y_max, x_min:x_max]


def crop_by_1point(image, point, target_shape):
    """
    Crop the image based on the starting point and specified width and height
    Args:
        image: The image to be cropped
        point: Top left corner coordinates (x, y) of the cropping area
        target_shape (tuple): A tuple (width, height) specifying the target size.

    Returns: Cropped image
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("np_img should be a NumPy array. Got {}.".format(type(image)))
    h, w = image.shape[:2]
    x_start, y_start = point
    width, height = to_2tuple(target_shape)
    assert width > 0 and height > 0, "Width and height of cropping area must be greater than 0"
    assert x_start >= 0 and y_start >= 0, "x_min and y_min cannot be less than 0"
    assert x_start + width <= w, f"Cropping out of bounds: x_start({x_start}) + width({width}) > image_width({w})"
    assert y_start + height <= h, f"Cropping out of bounds: y_start({y_start}) + height({height}) > image_height({h})"

    return image[y_start:y_start+height, x_start:x_start+width]


@_sync_wrapper
def center_crop(imglists, target_shape, is_labels=None):
    """
    Center-crops an image to the specified target size.
    If the image is smaller than the target size, it will be resized proportionally first.

    Args:
        imglists (np.ndarray or list): The input image(s).
        target_shape (tuple): A tuple (width, height) specifying the target size.
        is_labels (bool or list, optional): Whether the input is a label map.
            Labels use nearest-neighbor interpolation during resizing. Defaults to None.

    Returns:
        list: The center-cropped image(s).
    """
    imglists = ensure_min_size(imglists, target_shape, is_labels=is_labels)
    h, w = imglists[0].shape[:2]
    target_w, target_h = to_2tuple(target_shape)

    x_start = (w - target_w) // 2
    y_start = (h - target_h) // 2

    return [crop_by_1point(img, (x_start, y_start), target_shape) for img in imglists]



@_sync_wrapper
def random_crop(imglists, target_shape, is_labels=None):
    """
    Randomly crop an image to the specified size.
    If the image is smaller than the target size, it will be resized proportionally first.

    Args:
        imglists (np.ndarray or list): Input image(s).
        target_shape (tuple): A tuple (width, height) specifying the crop size.
        is_labels (bool or list, optional): Whether the input is a label map.
            Labels use nearest-neighbor interpolation during resizing. Defaults to None.

    Returns:
        list: Randomly cropped image(s).
    """
    imglists = ensure_min_size(imglists, target_shape, is_labels=is_labels)
    height, width = imglists[0].shape[:2]
    target_w, target_h = to_2tuple(target_shape)
    max_x = width - target_w
    max_y = height - target_h
    start_y = random.randint(0, max_y)
    start_x = random.randint(0, max_x)
    return [crop_by_1point(img, (start_x, start_y), target_shape) for img in imglists]


def five_crop(image, target_shape):
    """
    Generate 5 cropped images (one central and four corners).
    If the image is smaller than the target size, it will be resized proportionally first.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        target_shape (tuple): A tuple (width, height) specifying the crop size.

    Returns:
        list: A list of 5 NumPy arrays.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("np_img should be a NumPy array. Got {}.".format(type(image)))

    h, w = image.shape[:2]
    crop_w, crop_h = to_2tuple(target_shape)

    center_x = w // 2
    center_y = h // 2

    crops = []

    # Central crop
    left = center_x - crop_w // 2
    upper = center_y - crop_h // 2
    crops.append(crop_by_1point(image, (left, upper), (crop_w, crop_h)))

    # Top-left corner crop
    crops.append(crop_by_1point(image, (0, 0), (crop_w, crop_h)))

    # Top-right corner crop
    crops.append(crop_by_1point(image, (w - crop_w, 0), (crop_w, crop_h)))

    # Bottom-left corner crop
    crops.append(crop_by_1point(image, (0, h - crop_h), (crop_w, crop_h)))

    # Bottom-right corner crop
    crops.append(crop_by_1point(image, (w - crop_w, h - crop_h), (crop_w, crop_h)))

    return crops


# ----------------------------------Other----------------------------------
def erase(image, point, width, height, erase_value=None, is_label=False):
    """
    Erase a rectangular region in a NumPy image array.

    Args:
        image (numpy.ndarray): Input NumPy image array.
        point (tuple): Top left corner coordinates (x, y) of the cropping area
        width (int): Width of the erased region.
        height (int): Height of the erased region.
        erase_value (tuple or int, optional): The value to fill the erased region with.
            Defaults to (0, 0, 0) for labels and (128, 128, 128) for images.
        is_label (bool): Whether the input is a label map.

    Returns:
        numpy.ndarray: NumPy image array with the specified region erased and filled with the erase_value.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("np_img should be a NumPy array. Got {}.".format(type(image)))
    if erase_value is None:
        erase_value = (0, 0, 0) if is_label else (128, 128, 128)
    is_normalized = image.dtype == np.float32 and np.max(image) <= 1.0
    if is_normalized:
        erase_value = tuple(c / 255.0 for c in erase_value)
    x, y = point
    img_h, img_w = image.shape[:2]
    y_end = min(y + height, img_h)
    x_end = min(x + width, img_w)

    erased_img = image.copy()

    if len(erased_img.shape) == 3:
        erased_img[y:y_end, x:x_end, :] = erase_value
    else:
        erase_value = erase_value[0] if isinstance(erase_value, tuple) else erase_value
        erased_img[y:y_end, x:x_end] = erase_value[0] if isinstance(erase_value, tuple) else erase_value

    return erased_img

@_sync_wrapper
def random_erase(imglists, target_shape, erase_value=None, is_labels=None):
    """
    Randomly erase a rectangular region of a specified size in a NumPy image array.

    Args:
        imglists (np.ndarray or list): Input image(s).
        target_shape (tuple): A tuple (width, height) specifying the size of the erased region.
        erase_value (tuple or int, optional): The value to fill the erased region with.
            Defaults to (0, 0, 0) for labels and (128, 128, 128) for images.
        is_labels (bool or list, optional): Whether the input is a label map. Defaults to None.

    Returns:
        list: Image(s) with a random region erased.
    """
    is_labels = _parse_label_flags(imglists, is_labels)
    img_h, img_w = imglists[0].shape[:2]
    target_w, target_h = to_2tuple(target_shape)

    if target_w > img_w or target_h > img_h:
        return [img.copy() for img in imglists]

    x = random.randint(0, img_w - target_w)
    y = random.randint(0, img_h - target_h)

    result = []
    for i, img in enumerate(imglists):
        result.append(erase(img, point=(x, y), width=target_w, height=target_h,
                            erase_value=erase_value, is_label=is_labels[i]))
    return result

def stitcher_image(image_list: list):
    """
    Stitch a list of images into a single panoramic image.
    Compatible with both OpenCV 3.x and 4.x versions.

    Args:
        image_list (list): A list of NumPy arrays (images) to be stitched together.

    Returns:
        numpy.ndarray: The stitched panoramic image.

    Raises:
        AssertionError: If less than 2 images are provided, or if the stitching process fails.
    """
    opencv_version = cv2.__version__
    major_version = int(opencv_version.split(".")[0])

    if major_version >= 4:
        stitcher = cv2.Stitcher.create()  # OpenCV 4.x and above
    else:
        stitcher = cv2.createStitcher()   # OpenCV 3.x and earlier
    assert len(image_list) >= 2, "At least two images are required for stitching"
    (status, stitched_image) = stitcher.stitch(image_list)
    assert status == cv2.Stitcher_OK, 'Image stitching failed'

    return stitched_image


def translate(image, x, y):
    """
    Translate (shift) the input image by the specified x and y offsets.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        x (int): Number of pixels to shift the image along the x-axis (horizontal).
            Positive values shift the image to the right, negative values to the left.
        y (int): Number of pixels to shift the image along the y-axis (vertical).
            Positive values shift the image downward, negative values upward.

    Returns:
        numpy.ndarray: Translated (shifted) image.
    """
    shifted = cv2.warpAffine(
        image, np.float32([[1, 0, x], [0, 1, y]]), (image.shape[1], image.shape[0]))
    return shifted

@_sync_wrapper
def random_perspective(imglists, degrees: float = 30, translate: float = 0.1, scale: float = 0.1, is_labels = None) -> np.ndarray:
    """
    Apply a random affine transformation (rotation, scaling, and translation).

    Args:
        imglists (np.ndarray or list): Input image(s).
        degrees (float, optional): Maximum rotation angle in degrees.
            The actual angle will be randomly selected from [-degrees, degrees]. Default is 30.
        translate (float, optional): Maximum translation factor relative to image dimensions.
            The actual shift will be randomly selected from [-translate*dim, translate*dim]. Default is 0.1.
        scale (float, optional): Scale factor range for random scaling.
            The actual scale will be randomly selected from [1-scale, 1+scale]. Default is 0.1.
        is_labels (bool or list, optional): Whether the input is a label map.
            Labels use nearest-neighbor interpolation and black padding. Defaults to None.

    Returns:
        list: Transformed image(s) with the same dimensions as the input.
    """
    is_labels = _parse_label_flags(imglists, is_labels)
    height, width = imglists[0].shape[:2]

    angle = uniform(-degrees, degrees)
    scale_val = uniform(1 - scale, 1 + scale)
    tx = translate * width * uniform(-1, 1)
    ty = translate * height * uniform(-1, 1)

    M = cv2.getRotationMatrix2D((width/2, height/2), angle, scale_val)
    M[:, 2] += (tx, ty)

    result = []
    for i, img in enumerate(imglists):
        if is_labels[i]:
            flags = cv2.INTER_NEAREST
            borderMode = cv2.BORDER_CONSTANT
            borderValue = (0, 0, 0) if len(img.shape) == 3 else 0
        else:
            flags = cv2.INTER_LINEAR
            borderMode = cv2.BORDER_REFLECT
            borderValue = None

        transformed = cv2.warpAffine(img, M, (width, height), flags=flags, borderMode=borderMode, borderValue=borderValue)
        result.append(transformed)

    return result

if __name__=="__main__":
    image = cv2.imread(r"/resources/images/aircraft.jpg")
    label = cv2.imread(r'/resources/images/aircraft_mask.png')
    print(image.shape)
    image2 = None
    # ---------------------resize
    # image2 = smart_resize(image, (512, 512))
    # image2 = resize_samescale(image, width=512)
    # image2, pad_meta = resize_pad(image, (256, 256))
    # image2 = resize_unpad(image, pad_meta)
    # image2 = centerzoom(image, zoom_factor=2)
    # ---------------------flip & rotate
    # image2 = flip(image, 1)
    # image2 = random_flip(image, 2, prob=0.5)
    # image2 = rotate(image, 20, keep_bound=True)
    # image2 = random_rotate(image)
    # image2 = random_rot90(image)
    # ---------------------crop
    # image2 = crop_by_2point(image, (25, 24), (158, 245))
    # image2 = crop_by_1point(image, (25, 25), (256, 256))
    # image2 = center_crop(image, target_shape=(512, 512))
    image2 = random_crop(image, target_shape=(512, 512), is_labels=False)
    # image2 = five_crop(image, (128, 128))[1]
    # ---------------------other
    # image2 = erase(image, (52, 52), 25, 32, is_label=False)
    # image2 = random_erase(image, (52, 52), is_label=False)
    # image2 = stitcher_image([image, image])
    # image2 = translate(image, 24, 60)
    # image2 = random_perspective(image, 30, is_labels=False)
    # ---------------------test list
    # image2 = random_perspective([image, label], 50, is_labels=[False, True])
    image2 = ensure_min_size([image, label], (512, 512), [False, True])
    # image2 = random_flip([image, label])
    # image2 = random_crop([image, label], (256, 256), is_labels=[False, True])
    # image2 = random_rotate(image2, (15, 30), keep_bound=True, is_labels=[False, True])
    # image2 = random_rot90(image2)
    # image2 = center_crop(image2, (128, 128), is_labels=[False, True])
    # image2 = random_erase(image2, (56, 56), is_labels=[False, True])
    image2 = centerzoom(image2, 2, is_labels=[False, True])


    if isinstance(image2, list):
        print(image2[0].shape)
        cv2.imshow("image and label", np.hstack(image2))
    else:
        print(image2.shape)
        cv2.imshow("image", image2)
    cv2.waitKey(0)