"""
Copyright (c) 2023, Auorui.
All rights reserved.

Noise enhancement (JPEG compression, salt & pepper).
"""
import cv2
import numpy as np


def salt_pepper_noise(image: np.ndarray, prob=0.02) -> np.ndarray:
    """
    Add salt and pepper noise to the image.

    Args:
        image: Input image (numpy array)
        prob: Probability/intensity of salt & pepper noise (default: 0.02)

    Returns:
        Image with salt and pepper noise added
    """
    output = image.copy()
    h, w = image.shape[:2]
    noise = np.random.rand(h, w)

    salt_mask = noise < (prob / 2)
    pepper_mask = noise > (1 - prob / 2)

    if image.ndim == 2:
        output[salt_mask] = 255
        output[pepper_mask] = 0
    else:
        output[salt_mask] = [255] * image.shape[2]
        output[pepper_mask] = [0] * image.shape[2]

    return output


def add_jpeg_effect(image, quality=30):
    """
    Simulate JPEG compression artifacts (blocking and ringing effects).

    Args:
        image (numpy.ndarray): Input image in BGR format (uint8).
        quality (int): JPEG quality factor (1 to 100). Lower values produce more artifacts. Default is 30.

    Returns:
        numpy.ndarray: The JPEG-compressed and decompressed image, same shape and dtype as input.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("Input image must be a uint8 NumPy array.")
    if not 1 <= quality <= 100:
        raise ValueError("Quality must be between 1 and 100.")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded_image = cv2.imencode('.jpg', image, encode_param)

    if not success:
        raise RuntimeError("Failed to encode image to JPEG.")

    decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    return decoded_image


if __name__=="__main__":
    image_path = r'E:\PythonProject\pyzjrPyPi\resources\images\aircraft.jpg'
    image = cv2.imread(image_path)
    print(image.shape)
    # process_img = salt_pepper_noise(image)
    process_img = add_jpeg_effect(image)
    process_img = np.hstack([process_img, image])

    print(process_img.shape)
    cv2.imshow("img", process_img)
    cv2.waitKey(0)
