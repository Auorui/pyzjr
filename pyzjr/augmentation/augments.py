"""
这一部分主要针对于opencv格式的, 一般这部分用于传统图像的增强
"""
import numpy as np
import cv2
from math import ceil
import random
from pyzjr.core.error import _check_parameter_is_tuple_2, _check_parameter_is_tuple_and_list_2

__all__ = ["crop_image_by_points", "crop_image_by_dimensions", "center_crop", "five_crop", "Stitcher_image",\

           "Centerzoom", "flip", "horizontal_flip", "vertical_flip", "resize", "adjust_brightness_cv2", "adjust_brightness_numpy",\

           "rotate", "adjust_gamma", "pad", "erase", "augment_Hsv","hist_equalize","random_resize_crop","random_crop",\

           "random_horizontal_flip", "random_vertical_flip", "random_rotation", "random_lighting",\

           "Retinex"]

def crop_image_by_points(image, StartPoint, EndPoint):
    """
    根据起始点和结束点裁剪图像
    :param image: 要裁剪的图像
    :param StartPoint: 裁剪区域的左上角坐标
    :param EndPoint: 裁剪区域的右下角坐标
    :return: 被裁剪的图像
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

def crop_image_by_dimensions(image, StartPoint, width, height):
    """
    根据起始点和指定的宽度、高度裁剪图像
    :param image: 要裁剪的图像
    :param StartPoint: 裁剪区域的左上角坐标
    :param width: 裁剪区域的宽度
    :param height: 裁剪区域的高度
    :return: 被裁剪的图像
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

    return crop_image_by_dimensions(image, (x_start, y_start), crop_width, crop_height)

def five_crop(image, size):
    """
    Generate 5 cropped images (one central and four corners).

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        size (tuple): A tuple (width, height) specifying the crop size.

    Returns:
        list: A list of 5 NumPy arrays.
    """
    _check_parameter_is_tuple_2(size)

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

    return crops    # central_crop, top_left_crop, top_right_crop, bottom_left_crop, bottom_right_crop

def Stitcher_image(image_paths: list):
    """
    图像拼接，图片较小可能拼接失败
    :param image_paths: 由图片路径组成的列表
    :return: 返回被拼接好的图片
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
    assert len(images) >= 2, "[pyzjr]:At least two images are required for stitching"
    (status, stitched_image) = stitcher.stitch(images)
    assert status == cv2.Stitcher_OK, '[pyzjr]:Image stitching failed'
    return stitched_image

def Centerzoom(img, zoom_factor: int):
    """中心缩放"""
    h, w = img.shape[:2]
    h_ch, w_ch = ceil(h / zoom_factor), ceil(w / zoom_factor)
    h_top, w_top = (h - h_ch) // 2, (w - w_ch) // 2
    zoomed_img = cv2.resize(img[h_top : h_top + h_ch, w_top : w_top + w_ch], (w, h), interpolation=cv2.INTER_LINEAR)

    return zoomed_img


def flip(image, option_value):
    """
    :param image : numpy array of image
    :param option_value: random integer between 0 to 3
            vertical                          0
            horizontal                        1
            horizontally and vertically flip  2
    Return: image : numpy array of flipped image
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

def horizontal_flip(img):
    """Flip the image horizontally"""
    return flip(img, 1)

def vertical_flip(img):
    """Flip the image vertically"""
    return flip(img, 0)

def resize(image, size=None, scale=None):
    """
    双线性插值 https://blog.csdn.net/m0_62919535/article/details/132094815
    Args:
        image (numpy.ndarray): Image to be resized.
        size (tuple): New size in the format (width, height).
        scale (float): Scale of image
    Returns:
        Resized image.
    """
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

def adjust_gamma(img, gamma, gain=1):
    """Adjust gamma of the input opencv image."""
    img_np = np.array(img)
    img_gamma_corrected = ((img_np / 255.0) ** gamma) * 255.0 * gain
    img_gamma_corrected = np.clip(img_gamma_corrected, 0, 255).astype(np.uint8)
    return img_gamma_corrected

def pad(img, padding, fill_value=(128, 128, 128)):
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

    img_np = np.array(img)
    if img_np.shape[-1] == 1:
        fill_value = fill_value[0]
        # BORDER_CONSTANT  BORDER_ISOLATED
    img_np = cv2.copyMakeBorder(img_np, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_value)

    return img_np

def erase(np_img, x, y, height, width, erase_value=(128,128,128)):
    """
    Erase a rectangular region in a NumPy image array.

    Args:
        np_img (numpy.ndarray): Input NumPy image array.
        x (int): X-coordinate of the top-left corner of the region to be erased.
        y (int): Y-coordinate of the top-left corner of the region to be erased.
        height (int): Height of the erased region.
        width (int): Width of the erased region.
        erase_value (tuple, optional): The RGB color value to fill the erased region with.
            Default is (128, 128, 128), which corresponds to gray.

    Returns:
        numpy.ndarray: NumPy image array with the specified region erased and filled with the erase_value.
    """
    if not isinstance(np_img, np.ndarray):
        raise TypeError("np_img should be a NumPy array. Got {}.".format(type(np_img)))

    erased_img = np_img.copy()
    erased_img[y:y + height, x:x + width, :] = erase_value
    return erased_img

def augment_Hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV颜色空间增强
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        return cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)

def hist_equalize(im, clahe=True, is_bgr=True):
    # 均衡BGR图像“im”上的直方图，其形状为im(n，m，3)，范围为0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if is_bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if is_bgr else cv2.COLOR_YUV2RGB)


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
    _check_parameter_is_tuple_2(target_size)
    _check_parameter_is_tuple_2(scale_range)

    min_scale, max_scale = scale_range
    scale_factor = random.uniform(min_scale, max_scale)

    height, width = image.shape[:2]

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    resized_image = resize(image, (new_width, new_height))

    crop_x = random.randint(0, new_width - target_size[0])
    crop_y = random.randint(0, new_height - target_size[1])
    cropped_image = crop_image_by_dimensions(resized_image, (crop_x, crop_y), target_size[0], target_size[1])

    return cropped_image

def random_crop(image, crop_size):
    """
    Randomly crop an image to the specified size.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        crop_size (tuple): A tuple (width, height) specifying the crop size.

    Returns:
        Randomly cropped image.
    """
    _check_parameter_is_tuple_2(crop_size)
    height, width = image.shape[:2]
    max_x = width - crop_size[0]
    max_y = height - crop_size[1]

    start_x = random.randint(0, max_x)
    start_y = random.randint(0, max_y)

    cropped_image = crop_image_by_dimensions(image, (start_x, start_y), crop_size[0], crop_size[1])

    return cropped_image

def random_horizontal_flip(img, prob=0.5):
    """Randomly flip the input image horizontally."""
    if prob > random.random():
        img = horizontal_flip(img)
    return img

def random_vertical_flip(img, prob=0.5):
    """Randomly flip the input image vertically."""
    if prob > random.random():
        img = vertical_flip(img)
    return img

def random_rotation(img, degrees=None, fill_value=(0, 0, 0)):
    """Randomly rotate the image"""
    if degrees is not None:
        _check_parameter_is_tuple_and_list_2(degrees)
        angle = random.uniform(degrees[0], degrees[1])
    else:
        angle = random.choice([0, 90, 180, 270])

    return rotate(img, angle, fill_value)

def random_lighting(img, alpha):
    """Add AlexNet-style PCA-based noise to an image."""
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
    img_arr = np.array(img).astype(np.float64)
    img_arr[:, :, 0] += pca_b
    img_arr[:, :, 1] += pca_g
    img_arr[:, :, 2] += pca_r
    img_arr = np.uint8(np.minimum(np.maximum(img_arr, 0), 255))

    return img_arr

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
    import pyzjr
    image_path = r"test.png"
    img = pyzjr.imreader(image_path)
    img = Centerzoom(img, 0.5)
    print(pyzjr.improperties(img))
    pyzjr.display("test", img)