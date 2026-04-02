"""
Copyright (c) 2023, Auorui.
All rights reserved.

Geometric transformations based on OpenCV (rotation, scaling, cropping, resizing, filling, erasing, etc.)
"""
import cv2
import random
import warnings
import numpy as np
from math import ceil
from pyzjr.utils.randfun import rand

def crop_image_by_2points(image, StartPoint, EndPoint):
    """
    Crop the image based on the starting and ending points
    Args:
        image: The image to be cropped
        StartPoint: Top left corner coordinates (x, y) of the cropping area
        EndPoint: The coordinates of the bottom right corner of the cropping area (x1, y1)

    Returns:Cropped image
    """
    height, width = image.shape[:2]
    x_min, y_min = StartPoint
    x_max, y_max = EndPoint

    assert x_max > x_min, "Maximum value of cropping x_max cannot be less than the minimum value x_min"
    assert y_max > y_min, "Maximum value of cropping y_max cannot be less than the minimum value y_min"
    assert x_min >= 0, "x_min cannot be less than 0"
    assert x_max <= width, "x_max cannot be greater than the image width"
    assert y_min >= 0, "y_min cannot be less than 0"
    assert y_max <= height, "y_max cannot be greater than the image height"

    return image[y_min:y_max, x_min:x_max]

def crop_image_by_1points(image, StartPoint, target_shape):
    """
    Crop the image based on the starting point and specified width and height
    Args:
        image: The image to be cropped
        StartPoint: Top left corner coordinates (x, y) of the cropping area
        target_shape (tuple): A tuple (width, height) specifying the target size.

    Returns: Cropped image
    """
    x_start, y_start = StartPoint
    width, height = target_shape
    assert width > 0 and height > 0, "Width and height of cropping area must be greater than 0"
    assert x_start >= 0 and y_start >= 0, "x_min and y_min cannot be less than 0"

    return image[y_start:y_start+height, x_start:x_start+width]

def center_crop(image, target_shape):
    """
    Center-crops an image to the specified target size.

    Args:
        image (numpy.ndarray): The input image.
        target_shape (tuple): A tuple (width, height) specifying the target size.

    Returns:
        numpy.ndarray: The center-cropped image.
    """
    h, w = image.shape[:2]
    crop_width, crop_height = target_shape

    if crop_width > w or crop_height > h:
        raise ValueError("center_crop: Target size is larger than the input image size")

    x_start = (w - crop_width) // 2
    y_start = (h - crop_height) // 2

    return crop_image_by_1points(image, (x_start, y_start), target_shape)


def five_crop(image, target_shape):
    """
    Generate 5 cropped images (one central and four corners).

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        target_shape (tuple): A tuple (width, height) specifying the crop size.

    Returns:
        list: A list of 5 NumPy arrays.
    """
    width, height = image.shape[1], image.shape[0]
    crop_width, crop_height = target_shape

    if crop_width > width or crop_height > height:
        raise ValueError("Crop size exceeds the image dimensions.")

    center_x = width // 2
    center_y = height // 2

    crops = []

    # Central crop
    left = center_x - crop_width // 2
    upper = center_y - crop_height // 2
    right = left + crop_width
    lower = upper + crop_height
    central_crop = image[upper:lower, left:right]
    crops.append(central_crop)

    # Top-left corner crop
    top_left_crop = image[0:crop_height, 0:crop_width]
    crops.append(top_left_crop)

    # Top-right corner crop
    top_right_crop = image[0:crop_height, width - crop_width:width]
    crops.append(top_right_crop)

    # Bottom-left corner crop
    bottom_left_crop = image[height - crop_height:height, 0:crop_width]
    crops.append(bottom_left_crop)

    # Bottom-right corner crop
    bottom_right_crop = image[height - crop_height:height, width - crop_width:width]
    crops.append(bottom_right_crop)

    return crops

def stitcher_image(image_paths: list):
    """
    Image stitching, smaller images may result in stitching failure
    Args:
        image_paths: A list composed of image paths

    Returns: the stitched image
    """
    opencv_version = cv2.__version__
    major_version = int(opencv_version.split(".")[0])
    if major_version >= 4:
        stitcher = cv2.Stitcher.create() # 适用于OpenCV 4.x
    else:
        stitcher = cv2.createStitcher()  # 适用于OpenCV 3.x 或更早版本
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    assert len(images) >= 2, "At least two images are required for stitching"
    (status, stitched_image) = stitcher.stitch(images)
    assert status == cv2.Stitcher_OK, 'Image stitching failed'
    return stitched_image


def centerzoom(image, zoom_factor: float):
    """
    Center Zoom Image
    Args:
        image: The image to be scaled
        zoom_factor: Scale factor

    Returns: the image scaled by the center

    """
    h, w = image.shape[:2]
    h_ch, w_ch = ceil(h / zoom_factor), ceil(w / zoom_factor)
    h_top, w_top = (h - h_ch) // 2, (w - w_ch) // 2
    zoomed_img = cv2.resize(image[h_top : h_top + h_ch, w_top : w_top + w_ch], (w, h), interpolation=cv2.INTER_LINEAR)

    return zoomed_img


def flip(image, option_value):
    """
    Flip the image
    Args:
        image: numpy array of image
        option_value: random integer between 0 to 2
            vertical                          0
            horizontal                        1
            diagonal                          2
    Returns: numpy array of flipped image
    """
    if option_value == 0:
        image = np.flip(image, option_value)
    elif option_value == 1:
        image = np.flip(image, option_value)
    elif option_value == 2:
        image = np.flip(image, 0)
        image = np.flip(image, 1)
    else:
        image = image
    return image


def vertical_flip(image):
    """Flip the image vertically"""
    return flip(image, 0)

def horizontal_flip(image):
    """Flip the image horizontally"""
    return flip(image, 1)

def diagonal_flip(image):
    """Flip the image vertically"""
    return flip(image, 2)


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


def rotate(image, angle, fill_value=(0, 0, 0)):
    """
    Rotate the input image by angle degrees.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        angle (float): Rotation angle in degrees, counter-clockwise.
        fill_value (tuple, optional): Fill color for areas outside the rotated image.
            Default is (0, 0, 0) for black.

    Returns:
        numpy.ndarray: Rotated image.

    """
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=fill_value)
    return rotated_image

def rotate_bound(image, angle):
    """
    Rotate the input image by angle degrees without cropping.Ensure maximum boundary.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        angle (float): Rotation angle in degrees, counter-clockwise.

    Returns:
        numpy.ndarray: Rotated image with no cropping.

    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))

def resize(image, target_shape=None, scale=None):
    """
    bilinear interpolation: https://blog.csdn.net/m0_62919535/article/details/132094815
    Args:
        image (numpy.ndarray): Image to be resized.
        target_shape (tuple): New size in the format (width, height).
        scale (float): Scale of image
    Returns:
        Resized image.
    """
    if target_shape is None and scale is None:
        raise ValueError("Either size or scale must be provided.")

    if scale is not None:
        ah, aw, channel = image.shape
        bh, bw = int(ah * scale), int(aw * scale)
        dst_img = np.zeros((bh, bw, channel), np.uint8)

        y_coords, x_coords = np.meshgrid(np.arange(bh), np.arange(bw), indexing='ij')
        AX = (x_coords + 0.5) / scale - 0.5
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

    if target_shape is not None:
        dst_img = cv2.resize(image, target_shape, interpolation=cv2.INTER_LINEAR)

    return dst_img.astype(np.uint8)

def resize_samescale(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    """
    Resize the input image while maintaining the aspect ratio.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        width (int, optional): Target width of the resized image. If None, it will be calculated based on the height.
        height (int, optional): Target height of the resized image. If None, it will be calculated based on the width.
        interpolation (int, optional): Interpolation method for resizing. Default is cv2.INTER_AREA.

    Returns:
        numpy.ndarray: Resized image with the same aspect ratio as the input image.
    """
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    elif width is not None and height is not None:
        warnings.warn("Both width and height are specified. The image will be resized based on the width while maintaining the aspect ratio. "
                      "The specified height may not be exactly met.")
    dim = None
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=interpolation)
    return resized


def resizepad(image, target_shape, label=None, pad_color=(128, 128, 128)):
    """
    Adjust the image size and perform grayscale filling
    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        target_shape (tuple): A tuple (width, height) specifying the target shape.

    Returns:
        the adjusted image
    """
    w, h = target_shape
    ih, iw = image.shape[:2]
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    is_normalized = image.dtype == np.float32 and np.max(image) <= 1.0
    if is_normalized:
        pad_color = tuple(c / 255.0 for c in pad_color)
        output_dtype = np.float32
    else:
        output_dtype = np.uint8
    interp = cv2.INTER_LINEAR if scale < 1 else cv2.INTER_CUBIC
    resized_image = cv2.resize(image, (nw, nh), interpolation=interp)
    new_image = np.full((h, w, 3), pad_color, dtype=output_dtype)
    top = (h - nh) // 2
    left = (w - nw) // 2
    new_image[top:top + nh, left:left + nw] = resized_image
    if label is not None:
        resized_label = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST)
        if label.ndim == 3:
            new_label = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            new_label = np.zeros((h, w), dtype=np.uint8)
        new_label[top:top + nh, left:left + nw] = resized_label
        return new_image, new_label
    else:
        return new_image


def croppad_resize(image, original_shape, interpolation=cv2.INTER_LINEAR):
    """
    Crop the padded and resized image to get the closest view to the original aspect ratio
    and then resize it to the target shape.

    Args:
        image (numpy.ndarray): The image output from the resizepad function.
        original_shape (tuple): A tuple (height, width) specifying the original shape of the image.
        interpolation (int): The interpolation method to use for resizing.

    Returns:
        resized_image (numpy.ndarray): The final resized image.
    """
    h, w = image.shape[:2]
    ih, iw = original_shape
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    cropped_image = image[int((h - nh) // 2) : int((h - nh) // 2 + nh), \
                    int((w - nw) // 2) : int((w - nw) // 2 + nw)]
    resized_image = cv2.resize(cropped_image, (iw, ih), interpolation=interpolation)

    return resized_image

def random_rotation(image, degrees=None, fill_value=(0, 0, 0)):
    """
    Randomly rotate the image based on whether it is a square and the provided degree range.
    Args:
        image (numpy.ndarray):  The input image should be a NumPy array in BGR format.
        degrees (tuple, optional):  The range of rotation angles is in the format of (min_degree, max_degree). If None, select the default angle based on the shape of the image. The default is None.
        fill_value (tuple, optional):  The color filled in the blank area at the edge of the rotated image is in the format of (B, G, R). The default is (0,0,0), which is black.

    Returns:
        numpy.ndarray:  The rotated image is returned in BGR format.
    """
    h, w = image.shape[:2]
    is_squre = True if h == w else False
    if degrees is not None:
        angle = random.uniform(degrees[0], degrees[1])
    else:
        if is_squre:
            angle = random.choice([0, 90, 180, 270])
        else:
            angle = random.choice([0, 180])
    return rotate(image, angle, fill_value)

def random_rot90(image):
    """Randomly rotate the input image by 90 degrees"""
    r = random.randint(0, 3)
    return np.rot90(image, r, (0, 1))

def random_horizontal_flip(image, prob=0.5):
    """Randomly flip the input image horizontally."""
    if prob > random.random():
        image = horizontal_flip(image)
    return image

def random_vertical_flip(image, prob=0.5):
    """Randomly flip the input image vertically."""
    if prob > random.random():
        image = vertical_flip(image)
    return image

def random_crop(image, target_shape):
    """
    Randomly crop an image to the specified size.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        target_shape (tuple): A tuple (width, height) specifying the crop size.

    Returns:
        Randomly cropped image.
    """
    height, width = image.shape[:2]
    max_x = width - target_shape[0]
    max_y = height - target_shape[1]

    start_x = random.randint(0, max_x)
    start_y = random.randint(0, max_y)

    cropped_image = crop_image_by_1points(image, (start_x, start_y), target_shape)

    return cropped_image

def random_resize_crop(image, target_shape, scale_range=(1., 2.)):
    """
    Randomly resize and crop an image.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        target_shape (tuple): A tuple (width, height) specifying the target size.
        scale_range (tuple, optional): A tuple (min_scale, max_scale) specifying the range of scaling.
            Default is (0.8, 1.2), which allows resizing between 80% and 120% of the original size.

    Returns:
        Randomly resized and cropped image.
    """
    min_scale, max_scale = scale_range
    scale_factor = random.uniform(min_scale, max_scale)
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    resized_image = resize(image, (new_width, new_height))

    crop_x = random.randint(0, new_width - target_shape[0])
    crop_y = random.randint(0, new_height - target_shape[1])
    cropped_image = crop_image_by_1points(resized_image, (crop_x, crop_y), target_shape)

    return cropped_image

def pad_margin(image, padding, fill_value=(128, 128, 128)):
    """
    Add margins to the input image.
    Args:
        image: Input the image as a NumPy array.
        padding: Margin size.
            -If it is an integer, all margins (top, bottom, left, right) are set to that value.
            -If it is a tuple or list of length 2, the first element represents the left and right margins, and the second element represents the top and bottom margins.
            -If it is a tuple or list of length 4, it represents the left, top, right, and bottom margins in order.
        fill_value: Fill color values (BGR format)
    Returns: Image with added margins
    """
    if not isinstance(padding, (int, tuple, list)):
        raise TypeError("Padding should be an integer or a tuple/list of two or four values.")

    if not isinstance(fill_value, (int, str, tuple)):
        raise TypeError("Fill value should be an integer, a string, or a tuple.")

    top = bottom = left = right = None
    if isinstance(padding, int):
        top = bottom = left = right = padding
    elif len(padding) == 2:
        left = right = padding[0]
        top = bottom = padding[1]
    elif len(padding) == 4:
        left = padding[0]
        top = padding[1]
        right = padding[2]
        bottom = padding[3]

    img_np = np.array(image)
    if img_np.shape[-1] == 1:
        fill_value = fill_value[0]
        # BORDER_CONSTANT  BORDER_ISOLATED
    img_np = cv2.copyMakeBorder(img_np, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_value)

    return img_np

def erase(image, StartPoint, height, width, erase_value=(128,128,128)):
    """
    Erase a rectangular region in a NumPy image array.

    Args:
        image (numpy.ndarray): Input NumPy image array.
        StartPoint (tuple): Top left corner coordinates (x, y) of the cropping area
        x (int): X-coordinate of the top-left corner of the region to be erased.
        y (int): Y-coordinate of the top-left corner of the region to be erased.
        height (int): Height of the erased region.
        width (int): Width of the erased region.
        erase_value (tuple, optional): The RGB color value to fill the erased region with.
            Default is (128, 128, 128), which corresponds to gray.

    Returns:
        numpy.ndarray: NumPy image array with the specified region erased and filled with the erase_value.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("np_img should be a NumPy array. Got {}.".format(type(image)))
    x, y = StartPoint
    erased_img = image.copy()
    erased_img[y:y + height, x:x + width, :] = erase_value
    return erased_img

def random_perspective(image: np.ndarray, degrees: float = 10, translate: float = 0.1, scale: float = 0.1) -> np.ndarray:
    """Apply random perspective transformation.

    Args:
        image (numpy.ndarray): Input image array to be transformed.
        degrees (float, optional): Maximum rotation angle range in degrees.
            The actual rotation will be randomly selected between [-degrees, degrees]. Default is 10.
        translate (float, optional): Maximum translation factor relative to image dimensions.
            Translation will be randomly selected between [-translate*dim, translate*dim]. Default is 0.1.
        scale (float, optional): Scale factor range for random scaling.
            The actual scale will be randomly selected between [1-scale, 1+scale]. Default is 0.1.

    Returns:
        numpy.ndarray: Transformed image with the same dimensions as the input.
    """
    height, width = image.shape[:2]
    # Rotation and scale
    angle = rand(-degrees, degrees)
    scale_val = rand(1 - scale, 1 + scale)
    # Translation
    tx = translate * width * rand(-1, 1)
    ty = translate * height * rand(-1, 1)
    # Transformation matrix
    M = cv2.getRotationMatrix2D((width/2, height/2), angle, scale_val)
    M[:, 2] += (tx, ty)
    return cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

if __name__=="__main__":
    image_path = r"E:\PythonProject\pyzjrPyPi\resources\images\bird.jpg"
    image = cv2.imread(image_path)
    print(image.shape)
    # process_img = crop_image_by_2points(image, StartPoint=(200, 20), EndPoint=(400, 100))
    # process_img = crop_image_by_1points(image, StartPoint=(200, 20), target_shape=(100, 200))
    # process_img = center_crop(image, target_shape=(200, 200))
    # process_img = five_crop(image, (100, 100))[0]
    # process_img = centerzoom(image, 1.2)
    # process_img = flip(image, 0)
    # process_img = vertical_flip(image)
    # process_img = horizontal_flip(image)
    # process_img = resize(image, (200, 200), scale=1.2)
    # process_img = resize_samescale(image, None, 300)
    # process_img = translate(image, 100, 50)
    # process_img = rotate(image, 20)
    # process_img = rotate_bound(image, 20)
    # process_img = pad_margin(image, padding=(24, 25))
    # process_img = erase(image, (100, 100), 120, 130)
    # process_img = random_rotation(image)
    # process_img = random_rot90(image)
    # process_img = random_horizontal_flip(image)
    # process_img = random_vertical_flip(image)
    # process_img = random_crop(image, (250, 250))
    # process_img = random_resize_crop(image, (250, 250))
    # process_img = resizepad(image, (256, 256))
    # process_img = croppad_resize(process_img, (335, 500))
    process_img = random_perspective(image, degrees=70)
    print(process_img.shape)
    # process_img = np.hstack([process_img, image])
    print(process_img.shape)
    cv2.imshow("img", process_img)
    cv2.waitKey(0)
