"""
Copyright (c) 2023, Auorui.
All rights reserved.

OpenCV based noise enhancement (adding noise, fog, Gaussian, etc.)
"""
import cv2
import numpy as np


def salt_pepper_noise(image: np.ndarray, prob=0.02):
    """
    Add salt and pepper noise to the image.

    Args:
        image: Input image (numpy array)
        prob: Probability/intensity of salt & pepper noise (default: 0.02)

    Returns:
        Image with salt and pepper noise added
    """
    image = add_uniform_noise(image, prob * 0.51, vaule=255)
    image = add_uniform_noise(image, prob * 0.5, vaule=0)
    return image


def add_uniform_noise(image: np.ndarray, prob=0.05, vaule=255):
    """
    Add uniform noise to the image (used as helper for salt_pepper_noise).

    Args:
        image: Input image (numpy array)
        prob: Probability of noise pixels (default: 0.05)
        vaule: Intensity value for the noise (default: 255)

    Returns:
        Image with uniform noise added
    """
    h, w = image.shape[:2]
    noise = np.random.uniform(low=0.0, high=1.0, size=(h, w)).astype(dtype=np.float32)  # 产生高斯噪声
    mask = np.zeros(shape=(h, w), dtype=np.uint8) + vaule
    index = noise > prob
    mask = mask * (~index)
    output = image * index[:, :, np.newaxis] + mask[:, :, np.newaxis]
    output = np.clip(output, 0, 255)
    output = np.uint8(output)
    return output

def gaussian_noise(image, mean=0.1, sigma=25):
    """
    Add Gaussian noise to an image.

    Args:
        image (numpy.ndarray): Input image (grayscale or color in BGR/RGB format)
        mean (float, optional): Mean of the Gaussian distribution (default: 0.1)
        sigma (float, optional): Standard deviation of the Gaussian distribution,
                               controls noise intensity (default: 25)

    Returns:
        numpy.ndarray: Image with Gaussian noise added (same format as input)
    """
    img = image.astype(np.float32)
    noise = np.random.normal(mean, sigma, image.shape)
    img += noise
    noisy_image = np.clip(img, 0, 255).astype(np.uint8)
    return noisy_image


def addfog_with_channels(image, fog_intensity=0.5, fog_color_intensity=255):
    """
    Apply fog effects to the RGB channels of the image.
    Args:
        image: Input image (numpy array).
        fog_intensity: The intensity of fog (0 to 1).
        fog_color_intensity: The color intensity of fog (0 to 255) Not too small, it is recommended to be greater than 180
    """
    fog_intensity = np.clip(fog_intensity, 0, 1)
    fog_layer = np.ones_like(image) * fog_color_intensity
    fogged_image = cv2.addWeighted(image, 1 - fog_intensity, fog_layer, fog_intensity, 0)

    return fogged_image


def addfog_with_asm(image, beta=0.05, brightness=0.5):
    """
    The atmospheric scattering model(ASM) has effectively achieved the effect of adding haze to the input image.
    Args:
        image (numpy.ndarray): Input image (numpy array).
        beta (float, optional): Parameters for controlling haze effects The larger the beta value, the more pronounced the haze effect The default value is 0.05
        brightness (float, optional): The brightness value of haze The larger the value, the higher the overall brightness of the image The default is 0.5
    Returns:
        numpy.ndarray: The image after adding haze effect has a data type of uint8 and a range of 0-255.
    """
    img_f = image.astype(np.float32) / 255.0
    row, col, chs = image.shape
    size = np.sqrt(max(row, col))  # Atomization size
    center = (row // 2, col // 2)  # Atomization center
    y, x = np.ogrid[:row, :col]
    dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    d = -0.04 * dist + size
    td = np.exp(-beta * d)
    img_f = img_f * td[..., np.newaxis] + brightness * (1 - td[..., np.newaxis])
    img_f = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    return img_f

def add_poisson_noise(image):
    """
    Add Poisson noise to an image to simulate photon counting noise in imaging systems.

    Poisson noise follows the distribution: P(x) = λˣ * e^(-λ) / x!
    where x is the number of photons received and λ is the exposure level.

    Args:
        image (numpy.ndarray): Input grayscale image (2D array).

    Returns:
        numpy.ndarray: Image with Poisson noise added, maintaining original dtype.
    """
    max_value = np.max(image)
    image_float = image.astype(np.float64)
    scaled = image_float / max_value * 255.0
    noisy = np.random.poisson(scaled)
    rescaled = noisy / 255.0 * max_value
    noisy_image = np.clip(rescaled, 0, max_value).astype(image.dtype)
    return noisy_image


def add_rayleigh_noise(image, scale=1, mode='multiplicative'):
    """
    Add Rayleigh noise to an image with configurable mixing mode.

    Rayleigh distribution: f(x) = x * exp(-x²/(2*scale²)) / (scale² * sqrt(2π))
    Commonly used to model gradient magnitudes in ultrasound or radar imaging.

    Args:
        image (numpy.ndarray): Input grayscale image.
        scale (float): Scale parameter for Rayleigh distribution (default: 1).
        mode (str): Noise mixing mode - 'additive' or 'multiplicative' (default).

    Returns:
        numpy.ndarray: Noisy image with same dtype as input.

    Raises:
        TypeError: If input image has unsupported dtype.
        ValueError: If invalid mode is specified.
    """
    if np.issubdtype(image.dtype, np.integer):
        max_val = np.iinfo(image.dtype).max
    elif np.issubdtype(image.dtype, np.floating):
        max_val = 1.0
    else:
        raise TypeError("Unsupported image dtype")

    noise = np.random.rayleigh(scale, image.shape)
    image_float = image.astype(np.float64)

    if mode == 'additive':
        noisy = image_float + noise
    elif mode == 'multiplicative':
        noisy = image_float * noise
    else:
        raise ValueError("Mode must be either 'additive' or 'multiplicative'")

    return np.clip(noisy, 0, max_val).astype(image.dtype)


def add_gamma_noise(image, shape=0.9, scale=1, mode='multiplicative'):
    """
    Add Gamma-distributed noise to an image with flexible mixing options.

    Gamma distribution: f(x) = x^(shape-1) * exp(-x/scale) / (scale^shape * Γ(shape))
    Useful for modeling speckle noise in SAR images or medical ultrasound.

    Args:
        image (numpy.ndarray): Input grayscale image.
        shape (float): Shape parameter (0.5-2.0 recommended range, default: 0.9).
        scale (float): Scale parameter (default: 1).
        mode (str): Noise mixing mode - 'additive' or 'multiplicative' (default).

    Returns:
        numpy.ndarray: Noisy image with original dtype.

    Raises:
        TypeError: For unsupported image dtypes.
        ValueError: For invalid mode selection.
    """
    if np.issubdtype(image.dtype, np.integer):
        max_val = np.iinfo(image.dtype).max
    elif np.issubdtype(image.dtype, np.floating):
        max_val = 1.0
    else:
        raise TypeError("Unsupported image dtype")

    gamma_noise = np.random.gamma(shape, scale, image.shape)
    image_float = image.astype(np.float64)

    if mode == 'additive':
        noisy = image_float + gamma_noise
    elif mode == 'multiplicative':
        noisy = image_float * gamma_noise
    else:
        raise ValueError("Mode must be 'additive' or 'multiplicative'")

    return np.clip(noisy, 0, max_val).astype(image.dtype)

if __name__=="__main__":
    image_path = r'E:\PythonProject\pyzjrPyPi\pyzjr\utils\tryout\images\bird.jpg'
    image = cv2.imread(image_path)
    print(image.shape)
    # process_img = salt_pepper_noise(image)
    # process_img = gaussian_noise(image)
    process_img = add_uniform_noise(image)
    # process_img = addfog_with_channels(image)
    # process_img = addfog_with_asm(image)
    # process_img = add_poisson_noise(image)
    # process_img = add_rayleigh_noise(image, scale=1)
    # process_img = add_gamma_noise(image, scale=1)
    process_img = np.hstack([process_img, image])

    print(process_img.shape)
    cv2.imshow("img", process_img)
    cv2.waitKey(0)
