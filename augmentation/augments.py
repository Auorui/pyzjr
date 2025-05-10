"""
Copyright (c) 2024, Auorui.
All rights reserved.

主要针对于opencv格式的, 一般这部分用于传统图像的增强
"""
import cv2
import random
import warnings
import numpy as np
from math import ceil

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


def crop_image_by_1points(image, StartPoint, width, height):
    """
    Crop the image based on the starting point and specified width and height
    Args:
        image: The image to be cropped
        StartPoint: Top left corner coordinates (x, y) of the cropping area
        width: Width of cropping area
        height: Height of cropping area

    Returns: Cropped image
    """
    x_start, y_start = StartPoint
    assert width > 0 and height > 0, "Width and height of cropping area must be greater than 0"
    assert x_start >= 0 and y_start >= 0, "x_min and y_min cannot be less than 0"

    return image[y_start:y_start+height, x_start:x_start+width]

def center_crop(image, target_size):
    """
    Center-crops an image to the specified target size.

    Args:
        image (numpy.ndarray): The input image.
        target_size (tuple): A tuple (width, height) specifying the target size.

    Returns:
        numpy.ndarray: The center-cropped image.
    """
    h, w = image.shape[:2]
    crop_width, crop_height = target_size

    if crop_width > w or crop_height > h:
        raise ValueError("center_crop: Target size is larger than the input image size")

    x_start = (w - crop_width) // 2
    y_start = (h - crop_height) // 2

    return crop_image_by_1points(image, (x_start, y_start), crop_width, crop_height)

def five_crop(image, size):
    """
    Generate 5 cropped images (one central and four corners).

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        size (tuple): A tuple (width, height) specifying the crop size.

    Returns:
        list: A list of 5 NumPy arrays.
    """
    width, height = image.shape[1], image.shape[0]
    crop_width, crop_height = size

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
            horizontally and vertically flip  2
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

def horizontal_flip(image):
    """Flip the image horizontally"""
    return flip(image, 1)

def vertical_flip(image):
    """Flip the image vertically"""
    return flip(image, 0)

def resize(image, size=None, scale=None):
    """
    bilinear interpolation: https://blog.csdn.net/m0_62919535/article/details/132094815
    Args:
        image (numpy.ndarray): Image to be resized.
        size (tuple): New size in the format (width, height).
        scale (float): Scale of image
    Returns:
        Resized image.
    """
    if size is None and scale is None:
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

    if size is not None:
        dst_img = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

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
        target_shape (tuple): A tuple (height, width) specifying the target shape.

    Returns:
        the adjusted image
    """
    h, w = target_shape
    ih, iw = image.shape[:2]
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_image = np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)
    top = (h - nh) // 2
    left = (w - nw) // 2
    new_image[top:top + nh, left:left + nw] = resized_image
    if label is not None:
        resized_label = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST)
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

def adjust_brightness_cv2(image, brightness_factor):
    """
    Adjust brightness of an image using OpenCV.
    Args:
        image (numpy.ndarray): Image to be adjusted.
        brightness_factor (float): A factor by which to adjust brightness.
            - 0.0 gives a black image.
            - 1.0 gives the original image.
            - Greater than 1.0 increases brightness.
            - Less than 1.0 decreases brightness.
    Returns:
        Brightness-adjusted image.
    """
    return cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

def adjust_brightness_numpy(image, brightness_factor):
    """
    Adjust brightness of an image using Numpy.
    Args:
        image (numpy.ndarray): Image to be adjusted.
        brightness_factor (float): A factor by which to adjust brightness.
            - 0.0 gives a black image.
            - 1.0 gives the original image.
            - Greater than 1.0 increases brightness.
            - Less than 1.0 decreases brightness.
    Returns:
        Brightness-adjusted image.
    """
    image_float = image.astype(np.float32)
    _image = image_float * brightness_factor
    _image = np.clip(_image, 0, 255)
    b_image = _image.astype(np.uint8)
    return b_image

def adjust_brightness_contrast(image, brightness=0., contrast=0.):
    """
    Adjust the brightness and/or contrast of an image.

    Args:
        image (numpy.ndarray): Input image as a NumPy array (OpenCV BGR image).
        brightness (float, optional): Brightness adjustment value.
            0 means no change, positive values increase brightness, and negative values decrease brightness.
        contrast (float, optional): Contrast adjustment value.
            0 means no change, positive values increase contrast, and negative values decrease contrast.

    Returns:
        numpy.ndarray: Image with adjusted brightness and contrast.

    """
    beta = 0
    return cv2.addWeighted(image,
                           1 + float(contrast) / 100.,
                           image,
                           beta,
                           float(brightness))

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

def adjust_gamma(image, gamma, gain=1):
    """
    Adjust the gamma correction and gain of an input OpenCV image.

    Gamma correction is a nonlinear operation used to encode or decode luminance
    or tristimulus values in video or still image systems. Increasing gamma
    values make the dark parts of an image darker and the bright parts brighter,
    while decreasing gamma values have the opposite effect. The gain parameter
    scales the intensity of the image after gamma correction.

    Args:
        image (numpy.ndarray): Input image as a NumPy array, typically in the range [0, 255]
            for 8-bit images.
        gamma (float): Gamma correction factor. Values typically range from 0.5 to 2.5.
            A value of 1.0 leaves the image unchanged.
        gain (float, optional): Gain factor to scale the image after gamma correction.
            Defaults to 1.0 (no change in intensity).

    Returns:
        numpy.ndarray: Gamma-corrected and gain-adjusted image as a NumPy array, with
            pixel values in the range [0, 255].
    """
    img_np = np.array(image)
    img_gamma_corrected = ((img_np / 255.0) ** gamma) * 255.0 * gain
    img_gamma_corrected = np.clip(img_gamma_corrected, 0, 255).astype(np.uint8)
    return img_gamma_corrected

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

def enhance_hsv(image, hgain=0.5, sgain=0.5, vgain=0.5):
    """
    HSV color space enhancement
    Args:
        image (numpy.ndarray):  The input image should be a NumPy array in BGR format.
        Hgain (float): The gain value of Hue (color tone). The default value is 0.5.
        Sgain (float): The gain value of saturation. The default value is 0.5.
        Vgain (float): The gain value of Value (brightness). The default value is 0.5.
            -The gain value can be any real number, but is usually set between -1 and 1.
            -Positive values will enhance the corresponding color components, while negative values will weaken.

    Returns: Enhanced HSV color space image, returned in BGR format
    """
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        dtype = image.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        return cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=image)

def hist_equalize(image, clahe=True, is_bgr=True):
    """
    Perform histogram equalization on the brightness channels of the image to enhance its contrast.
    You can choose to use regular histogram equalization (cv2. equalizeHist) or contrast limited adaptive histogram equalization (CLAHE, cv2. create CLAHE).

    Args:
        image (numpy.ndarray):  The input image should be a NumPy array in BGR or RGB format.
        clahe (bool):  Should we use Contrast Constrained Adaptive Histogram Equalization (CLAHE). The default is True.
                    -If True, use CLAHE for equalization.
                    -If False, use regular histogram equalization.
        is_bgr (bool):  Is the input image in BGR format. The default is True.
                    -If True, assume the input image is in BGR format and convert the color space accordingly.
                    -If False, assume the input image is in RGB format and convert the color space accordingly.

    Returns:
        numpy.ndarray: The image obtained by histogram equalization of the brightness
        channel is returned in BGR or RGB format, depending on the format of the input image.
    """
    # 均衡BGR图像“im”上的直方图，其形状为im(n，m，3)，范围为0-255
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV if is_bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if is_bgr else cv2.COLOR_YUV2RGB)


def random_lighting(image, alpha):
    """
    Add AlexNet-style PCA-based noise to an image.
    Args:
        image (numpy.ndarray):  The input image should be a NumPy array in BGR format.
        alpha:(float):  Control the standard deviation of noise amplitude.
                        The larger the alpha value, the greater the added noise.

    Returns:
        numpy.ndarray:  The image with PCA noise added is returned in BGR format with data type uint8.
    """
    alpha_b = np.random.normal(loc=0.0, scale=alpha)
    alpha_g = np.random.normal(loc=0.0, scale=alpha)
    alpha_r = np.random.normal(loc=0.0, scale=alpha)
    table = np.array([
        [55.46 * -0.5675, 4.794 * 0.7192, 1.148 * 0.4009],
        [55.46 * -0.5808, 4.794 * -0.0045, 1.148 * -0.8140],
        [55.46 * -0.5836, 4.794 * -0.6948, 1.148 * 0.4203]
    ])
    pca_b = table[2][0] * alpha_r + table[2][1] * alpha_g + table[2][2] * alpha_b
    pca_g = table[1][0] * alpha_r + table[1][1] * alpha_g + table[1][2] * alpha_b
    pca_r = table[0][0] * alpha_r + table[0][1] * alpha_g + table[0][2] * alpha_b
    img_arr = np.array(image).astype(np.float64)
    img_arr[:, :, 0] += pca_b
    img_arr[:, :, 1] += pca_g
    img_arr[:, :, 2] += pca_r
    img_arr = np.uint8(np.minimum(np.maximum(img_arr, 0), 255))

    return img_arr

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

def random_crop(image, crop_size):
    """
    Randomly crop an image to the specified size.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        crop_size (tuple): A tuple (width, height) specifying the crop size.

    Returns:
        Randomly cropped image.
    """
    height, width = image.shape[:2]
    max_x = width - crop_size[0]
    max_y = height - crop_size[1]

    start_x = random.randint(0, max_x)
    start_y = random.randint(0, max_y)

    cropped_image = crop_image_by_1points(image, (start_x, start_y), crop_size[0], crop_size[1])

    return cropped_image


def random_resize_crop(image, target_size, scale_range=(1., 2.)):
    """
    Randomly resize and crop an image.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        target_size (tuple): A tuple (width, height) specifying the target size.
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

    crop_x = random.randint(0, new_width - target_size[0])
    crop_y = random.randint(0, new_height - target_size[1])
    cropped_image = crop_image_by_1points(resized_image, (crop_x, crop_y), target_size[0], target_size[1])

    return cropped_image

def addnoisy(image, n=10000):
    """
    Add salt and pepper noise to the image.

    Args:
        image (numpy.ndarray):  The input image should be a NumPy array of grayscale or color images (BGR or RGB format).
        n (int, optional):  The number of salt and pepper noise points to be added. The default is 10000.

    Returns: The image after adding salt and pepper noise has the same format as the input image.
    """
    result = image.copy()
    h, w = image.shape[:2]
    for i in range(n):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        if np.random.randint(0, 2) == 0:
            result[x, y] = 0
        else:
            result[x, y] = 255
    return result

def addfog(image, beta=0.05, brightness=0.5, use_efficient=True):
    """
    Efficient implementation of adding haze effect to input images based on atmospheric scattering model.

    Args:
        image (numpy.ndarray): The input image ranges from 0-255
        beta (float, optional): Parameters for controlling haze effects The larger the beta
            value, the more pronounced the haze effect The default value is 0.05
        brightness (float, optional): The brightness value of haze The larger the value,
            the higher the overall brightness of the image The default is 0.5
        use_efficient (bool, optional): Whether to adopt a more efficient approach, default to True
    Returns:
        numpy.ndarray: Image with haze effect added, data type uint8, range 0-255。
    """
    img_f = image.astype(np.float32) / 255.0
    row, col, chs = image.shape
    size = np.sqrt(max(row, col))  # Atomization size
    center = (row // 2, col // 2)  # Atomization center
    if use_efficient:
        y, x = np.ogrid[:row, :col]
        dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        d = -0.04 * dist + size
        td = np.exp(-beta * d)
        img_f = img_f * td[..., np.newaxis] + brightness * (1 - td[..., np.newaxis])
    else:
        for j in range(row):
            for l in range(col):
                d = -0.04 * np.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
                td = np.exp(-beta * d)
                img_f[j][l][:] = img_f[j][l][:] * td + brightness * (1 - td)
    img_f = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    return img_f

def addfog_channels(image, fog_intensity=0.5, fog_color_intensity=255):
    """
    对图像 RGB 通道应用雾效。

    参数:
        image: 输入图像（numpy数组）。
        fog_intensity: 雾的强度（0到1）。
        fog_color_intensity: 雾的颜色强度（0到255）.不宜过小, 建议大于180
    """
    fog_intensity = np.clip(fog_intensity, 0, 1)
    fog_layer = np.ones_like(image) * fog_color_intensity
    fogged_image = cv2.addWeighted(image, 1 - fog_intensity, fog_layer, fog_intensity, 0)

    return fogged_image

class Retinex():
    """
    增强算法Retinex
    - 单尺度 : SSR
    - 多尺度 : MSR
    - 多尺度自适应增益 : MSRCR
    https://blog.csdn.net/m0_62919535/article/details/130372571
    """
    def SSR(self, img, sigma):
        """
        将输入图像转换为了对数空间。/255将像素值归一化到0到1之间，np.log1p取对数并加1是为了避免出现对数运算中分母为0的情况。二维离散傅里叶变换将
        图像从空间域变换到频率域，可以提取出图像中的频率信息。G_recs是用于计算高斯核的半径，result用于最后三通道的叠加。然后循环用于计算加权后的
        频率域图像，再逆二维离散傅里叶变换，得到反射图像，对反射图像进行指数变换，得到最终的输出图像。
        :param img: 输入图像
        :param sigma: 高斯分布的标准差
        :return:
        """
        img_log = np.log1p(np.array(img, dtype="float") / 255)
        img_fft = np.fft.fft2(img_log)
        G_recs = sigma // 2 + 1
        result = np.zeros_like(img_fft)
        rows, cols, deep = img_fft.shape
        for z in range(deep):
            for i in range(rows):
                for j in range(cols):
                    for k in range(1, G_recs):
                        G = np.exp(-((np.log(k) - np.log(sigma)) ** 2) / (2 * np.log(2) ** 2))
                        #计算高斯滤波器的权值，其中sigma是高斯分布的标准差，k是高斯滤波器的半径，G是高斯滤波器在该点的权值。
                        result[i, j] += G * img_fft[i, j]
        img_ssr = np.real(np.fft.ifft2(result))
        img_ssr = np.exp(img_ssr) - 1
        img_ssr = np.uint8(cv2.normalize(img_ssr, None, 0, 255, cv2.NORM_MINMAX))
        return img_ssr

    def MSR(self, img, scales):
        """
        MSR算法在图像增强中与SSR不同的是，它不需要进行频域变换，它主要是基于图像在多个尺度下的平滑处理和差分处理来提取图像的局部对比度信息和全
        局对比度信息，从而实现对图像的增强。
        在 MSR 算法中，先对图像进行对数变换得到对数图像，然后在不同的尺度下，使用高斯滤波对图像进行平滑处理，得到不同尺度下的平滑图像。接着，通
        过将对数图像和不同尺度下的平滑图像进行差分，得到多个尺度下的细节图像。最后，将这些细节图像加权融合，输出最终的增强图像。

        :param img:
        :param scales: 取值大概在1-10之间
        :return:
        """
        img_log = np.log1p(np.array(img, dtype="float") / 255)
        result = np.zeros_like(img_log)
        img_light = np.zeros_like(img_log)
        r, c, deep = img_log.shape
        for z in range(deep):
            for scale in scales:
                kernel_size = scale * 4 + 1
                # 高斯滤波器的大小，经验公式kernel_size = scale * 4 + cat
                sigma = scale
                img_smooth = cv2.GaussianBlur(img_log[:, :, z], (kernel_size, kernel_size), sigma)
                img_detail = img_log[:, :, z] - img_smooth
                result[:, :, z] += cv2.resize(img_detail, (c, r))
                img_light[:, :, z] += cv2.resize(img_smooth, (c, r))
        img_msr = np.exp(result+img_light) - 1
        img_msr = np.uint8(cv2.normalize(img_msr, None, 0, 255, cv2.NORM_MINMAX))
        return img_msr


    def MSRCR(self, img, scales, k):
        """
        使用 MSRCR 算法的步骤：
            1. 将输入图像转换为对数空间（log-space）。
            2. 对图像在不同尺度下进行平滑处理，得到不同尺度下的平滑图像。
            3. 计算细节图像，通过将对数图像与平滑图像相减获得。
            4. 对细节图像进行缩放，以匹配输入图像的大小。
            5. 根据尺度因子调整细节图像的权重。
            6. 将权重调整后的细节图像与平滑图像相加，得到增强的图像。
            7. 最后，将增强的图像反转对数变换，以还原原始图像的像素值范围。
        :param img:
        :param scales:取值大概在1-10之间
        :param k: k的取值范围在10~20之间比较合适。当k取值较小时，图像的细节增强效果比较明显，但会出现较强的噪点，当k取值较大时，图像的细节
                    增强效果不明显，但噪点会减少。
        :return:
        """
        img_log = np.log1p(np.array(img, dtype="float") / 255)
        result = np.zeros_like(img_log)
        img_light = np.zeros_like(img_log)
        r, c, deep = img_log.shape
        for z in range(deep):
            for scale in scales:
                kernel_size = scale * 4 + 1
                # 高斯滤波器的大小，经验公式kernel_size = scale * 4 + cat
                sigma = scale
                G_ratio=sigma**2/(sigma**2+k)
                img_smooth = cv2.GaussianBlur(img_log[:, :, z], (kernel_size, kernel_size), sigma)
                img_detail = img_log[:, :, z] - img_smooth
                result[:, :, z] += cv2.resize(img_detail, (c, r))
                result[:, :, z]=result[:, :, z]*G_ratio
                img_light[:, :, z] += cv2.resize(img_smooth, (c, r))

        img_msrcr = np.exp(result+img_light) - 1
        img_msrcr = np.uint8(cv2.normalize(img_msrcr, None, 0, 255, cv2.NORM_MINMAX))
        return img_msrcr

if __name__=="__main__":
    image_path = r"E:\PythonProject\Pytorch_Segmentation_Auorui\data\VOCdevkit\VOC2012\JPEGImages\2007_000042.jpg"
    image = cv2.imread(image_path)
    print(image.shape)
    # process_img = crop_image_by_2points(image, StartPoint=(200, 20), EndPoint=(400, 100))
    # process_img = crop_image_by_1points(image, StartPoint=(200, 20), width=100, height=200)
    # process_img = center_crop(image, target_size=(200, 200))
    # process_img = five_crop(image, (100, 100))[0]
    # process_img = centerzoom(image, 1.2)
    # process_img = flip(image, 0)
    # process_img = vertical_flip(image)
    # process_img = horizontal_flip(image)
    # process_img = resize(image, (200, 200), scale=1.2)
    # process_img = resize_samescale(image, None, 300)
    # process_img = translate(image, 100, 50)
    # process_img = adjust_brightness_cv2(image, 0.5)
    # process_img = adjust_brightness_numpy(image, 0.5)
    # process_img = adjust_brightness_contrast(image, 10, 20)
    # process_img = rotate(image, 20)
    # process_img = rotate_bound(image, 20)
    # process_img = adjust_gamma(image, 1.2, 1.2)
    # process_img = pad_margin(image, padding=(24, 25))
    # process_img = erase(image, (100, 100), 120, 130)
    # process_img = enhance_hsv(image)
    # process_img = hist_equalize(image, clahe=False)
    # process_img = random_lighting(image, alpha=0.7)
    # process_img = random_rotation(image)
    # process_img = random_rot90(image)
    # process_img = random_horizontal_flip(image)
    # process_img = random_vertical_flip(image)
    # process_img = random_crop(image, (250, 250))
    # process_img = random_resize_crop(image, (250, 250))
    process_img = resizepad(image, (256, 256))
    process_img = croppad_resize(process_img, (335, 500))
    process_img = np.hstack([process_img, image])
    # process_img = addnoisy(image, n=10000)
    # process_img = addfog(image)
    # process_img = addfog_channels(image)
    print(process_img.shape)
    cv2.imshow("img", process_img)
    cv2.waitKey(0)
