"""
Copyright (c) 2023, Auorui.
All rights reserved.

Color transformation based on OpenCV (brightness, contrast, histogram equalization, etc.)
"""
import cv2
import random
import numpy as np

def adjust_brightness_with_opencv(image, brightness_factor):
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


def adjust_brightness_with_numpy(image, brightness_factor):
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
    else:
        return image.copy()

def random_hsv(image):
    """
    Apply random perturbations to an image in HSV color space.

    Performs three independent random transformations:
    1. Hue shift (random value between -30 and 30 degrees)
    2. Saturation scaling (random factor between 0.5x and 1.5x)
    3. Value (brightness) scaling (random factor between 0.5x and 1.5x)

    Args:
        image (numpy.ndarray): Input BGR image (uint8 format, shape: H×W×3)

    Returns:
        numpy.ndarray: Transformed BGR image with same shape/dtype as input
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_channel, s_channel, v_channel = cv2.split(hsv_image)
    delta_h = random.randint(-30, 30)
    h_channel = (h_channel + delta_h) % 180
    s_scale = np.random.uniform(0.5, 1.5)
    v_scale = random.uniform(0.5, 1.5)
    s_channel = np.clip(s_channel * s_scale, 0, 255).astype(np.uint8)
    v_channel = np.clip(v_channel * v_scale, 0, 255).astype(np.uint8)
    hsv_image[..., 0] = h_channel.astype(np.uint8)
    hsv_image[..., 1] = s_channel
    hsv_image[..., 2] = v_channel
    new_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return new_image

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

def random_pca_lighting(image, alpha):
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

def color_jitter(
        image: np.ndarray,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
) -> np.ndarray:
    """Apply random color jitter to an image (brightness, contrast, saturation, hue).

    Args:
        image (np.ndarray): Input image in BGR format (HWC shape), expected to be:
            - float32 in range [0, 1] (preferred)
            - or uint8 in range [0, 255] (will be converted to float32)
        brightness (float): Max absolute brightness delta (applied additively).
        contrast (float): Max contrast scaling factor (applied multiplicatively).
        saturation (float): Max saturation scaling factor (applied in HSV space).
        hue (float): Max hue shift (in radians, applied in HSV space).

    Returns:
        np.ndarray: Color-jittered image in the same format and range as input.

    Note:
        - For float32 inputs, output remains in [0, 1].
        - For uint8 inputs, output remains in [0, 255].
    """
    if image.dtype == np.float32 and np.max(image) <= 1.0:
        raise ValueError("Input image appears to be already normalized (values in [0,1]). "
                         "Please provide image in [0,255] range.")
    # Ensure image is float32 in [0, 1] for consistent processing
    image = image.astype(np.float32) / 255.0

    # Brightness adjustment (additive)
    if brightness > 0:
        delta = np.random.uniform(-brightness, brightness)
        image = np.clip(image + delta, 0, 1)

    # Contrast adjustment (multiplicative)
    if contrast > 0:
        alpha = 1.0 + np.random.uniform(-contrast, contrast)
        image = np.clip(alpha * image, 0, 1)

    # Saturation & Hue adjustment (HSV space)
    if saturation > 0 or hue > 0:
        # Convert to HSV (OpenCV expects float32 in [0, 1] for H, [0, 255] for S/V)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Saturation scaling
        if saturation > 0:
            alpha = 1.0 + np.random.uniform(-saturation, saturation)
            s = np.clip(alpha * s, 0, 1)

        # Hue shifting (OpenCV Hue range: [0, 180] for uint8)
        if hue > 0:
            delta = np.random.uniform(-hue, hue) * 180  # Convert radians to OpenCV Hue units
            h = (h * 180 + delta) % 180  # Scale float H to [0,180], then shift
            h = h / 180  # Scale back to [0,1]

        image = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    image = (image * 255).clip(0, 255).astype(np.uint8)
    return image


if __name__=="__main__":
    image_path = r"E:\PythonProject\pyzjrPyPi\pyzjr\utils\tryout\images\coffee.png"
    image = cv2.imread(image_path)
    print(image.shape)
    # process_img = adjust_brightness_with_opencv(image, 0.5)
    # process_img = adjust_brightness_with_numpy(image, 0.5)
    # process_img = adjust_brightness_contrast(image, 10, 20)
    # process_img = adjust_gamma(image, 1.2, 1.2)
    # process_img = random_hsv(image)
    # process_img = enhance_hsv(image)
    process_img = color_jitter(image)
    # process_img = hist_equalize(image, clahe=False)
    # process_img = random_pca_lighting(image, alpha=1.7)
    process_img = np.hstack([process_img, image])
    print(process_img.shape)
    cv2.imshow("img", process_img)
    cv2.waitKey(0)
