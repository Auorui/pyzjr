"""
Copyright (c) 2022, Auorui.
All rights reserved.

OpenCV based filtering or blurring
"""
import cv2
import numpy as np

def gaussian_blur(image, kernel_size=3, sigma=0):
    """
    Apply Gaussian blur to an image.

    Args:
        image (numpy.ndarray): Input image (grayscale or color).
        kernel_size (int): Size of the Gaussian kernel (width, height) - must be positive and odd.
        sigma (float): Standard deviation in X/Y directions. If 0, computed from kernel_size.

    Returns:
        numpy.ndarray: Blurred image with same shape and dtype as input.
    """
    if kernel_size % 2 == 0 or kernel_size <= 1:
        raise ValueError("kernel_size must be odd and greater than 1")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)

def median_blur(image, kernel_size=3):
    """
    Apply median filter for salt-and-pepper noise reduction.

    Args:
        image (numpy.ndarray): Input image (grayscale or color).
        kernel_size (int): Aperture size (must be odd and >1).

    Returns:
        numpy.ndarray: Filtered image.

    Raises:
        ValueError: If kernel_size is even or <=1.
    """
    if kernel_size % 2 == 0 or kernel_size <= 1:
        raise ValueError("kernel_size must be odd and greater than 1")
    return cv2.medianBlur(image, kernel_size)

def mean_blur(image, kernel_size=3):
    """
    Apply normalized box filter (mean blur) to an image.

    Args:
        image (numpy.ndarray): Input image (grayscale or color).
        kernel_size (int): Size of the averaging kernel (width, height).

    Returns:
        numpy.ndarray: Blurred image with same shape and dtype as input.
    """
    if kernel_size % 2 == 0 or kernel_size <= 1:
        raise ValueError("kernel_size must be odd and greater than 1")
    return cv2.blur(image, (kernel_size, kernel_size))

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Edge-preserving bilateral filter.

    Args:
        image (numpy.ndarray): Input image (grayscale or color).
        d (int): Diameter of pixel neighborhood.
        sigma_color (float): Filter sigma in color space.
        sigma_space (float): Filter sigma in coordinate space.

    Returns:
        numpy.ndarray: Filtered image.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def motion_blur(image, kernel_size=15, angle=0):
    """
    Simulate linear motion blur.

    Args:
        image (numpy.ndarray): Input image.
        kernel_size (int): Length of motion blur kernel (odd recommended).
        angle (float): Motion direction in degrees (0=horizontal).

    Returns:
        numpy.ndarray: Blurred image.
    """
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size//2, :] = 1.0
    kernel = cv2.warpAffine(kernel,
                            cv2.getRotationMatrix2D((kernel_size/2-0.5, kernel_size/2-0.5), angle, 1.0),
                            (kernel_size, kernel_size))
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def guided_filter(guide_img, input_img, radius=15, eps=1e-2):
    """
    Guided image filter for edge-preserving smoothing.

    This filter uses a guidance image to preserve edges while smoothing the input image.
    Commonly used for detail enhancement, noise reduction, and image matting.

    Args:
        guide_img (numpy.ndarray): Guidance image (grayscale, uint8 [0,255])
        input_img (numpy.ndarray): Input image to be filtered (same size as guide_img)
        radius (int): Radius of the filtering window (determines kernel size)
        eps (float): Regularization parameter (default: 0.01). Controls the
                    degree of smoothing (higher = more smoothing)

    Returns:
        numpy.ndarray: Filtered image (uint8 [0,255])
    """
    guide_img = guide_img.astype(np.float32) / 255.0
    input_img = input_img.astype(np.float32) / 255.0
    # Calculate kernel size from radius (2r+1 x 2r+1)
    kernel_size = (2 * radius + 1, 2 * radius + 1)
    mean_guide = cv2.blur(guide_img, kernel_size)
    mean_input = cv2.blur(input_img, kernel_size)
    mean_guide_sq = cv2.blur(guide_img*guide_img, kernel_size)
    mean_guide_input = cv2.blur(guide_img*input_img, kernel_size)
    var_guide = mean_guide_sq - mean_guide * mean_guide
    cov_guide_input = mean_guide_input - mean_guide * mean_input
    a = cov_guide_input / (var_guide + eps)
    b = mean_input - a * mean_guide
    mean_a = cv2.blur(a, kernel_size)
    mean_b = cv2.blur(b, kernel_size)
    output = mean_a * guide_img + mean_b
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    return output



if __name__=="__main__":
    image_path = r"E:\PythonProject\pyzjrPyPi\pyzjr\utils\tryout\images\lena.png"
    image = cv2.imread(image_path)
    print(image.shape)

    # process_img = gaussian_blur(image, kernel_size=5)
    # process_img = median_blur(image, kernel_size=5)
    # process_img = bilateral_filter(image, d=9)
    # process_img = mean_blur(image, kernel_size=5)
    process_img = motion_blur(image, kernel_size=15)
    # process_img = guided_filter(image, image, 16)
    process_img = np.hstack([process_img, image])

    print(process_img.shape)
    cv2.imshow("img", process_img)
    cv2.waitKey(0)



