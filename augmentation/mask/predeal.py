import cv2
import numpy as np
from PIL import Image
from pyzjr.utils.check import is_bool, is_gray_image

def uint2single(image):
    # 255 -> (0, 1)
    return np.float32(image / 255)

def single2uint(image):
    # (0, 1) -> 255
    return np.uint8((image.clip(0, 1) * 255.).round())

def clip(image, maxval=255, dtype=np.uint8):
    """
    Truncate the pixel values of the image to the specified range and perform data type conversion
    """
    np_image = np.array(image)
    return np.clip(np_image, 0, maxval).astype(dtype)

def unique(image):
    """Returns a list of unique pixel values in the input image"""
    np_image = np.array(image).astype(np.uint8)
    return np.unique(np_image)

def create_rectmask(image, up_left_coord=None, lower_right_coord=None, bboxes=None):
    """
    Create a rectangular mask
    :param image: Original image.
    :param up_left_coord: [x1, y1]
    :param lower_right_coord: [x2, y2]
    :param bboxes: [x1, y1, x2, y2]
    """
    image = np.array(image).astype(np.uint8)
    h, w = image.shape[:2]
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    if bboxes is None:
        x1, y1 = up_left_coord
        x2, y2 = lower_right_coord
    else:
        x1, y1, x2, y2 = bboxes
    mask[y1: y2, x1: x2] = 255

    if mask.ndim == 3 and mask.shape[-1] > 1:
        mask = mask[:, :, 0]
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return mask, masked_image


def binarization(image, min_value=127, max_value=255):
    """
    Convert image to binary image
    :param image: Original image. --- (BGR format)
    :param min_value: Minimum threshold, default to 127
    :param max_value: Maximum threshold, default to 225
    """
    np_image = np.array(image).astype(np.uint8)
    gray_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, min_value, max_value, cv2.THRESH_BINARY)
    return binary_image

def approximate(image, std=127.5, dtype=np.uint8):
    """
    Convert a single channel image into a binary image.
    """
    if not is_gray_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[image > std] = 255
    image[image < std] = 0
    image = image.astype(dtype)
    return image

def ceilfloor(image, dtype=np.uint8):
    """
    The pixel value of the input image is limited between the maximum value of 255 and the minimum value of 0
    """
    if not is_gray_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype(dtype)
    return image

def up_low(image, lower, upper, dtype=np.uint8):
    """
    Create a binarized mask based on the given color range (lower and upper limits)
    :param lower: Lower limit of color range, (B, G, R)
    :param upper: Upper limit of color range, (B, G, R)
    """
    np_image = np.array(image).astype(dtype)
    lower = np.array(lower, dtype=dtype)
    upper = np.array(upper, dtype=dtype)
    _mask = cv2.inRange(np_image, lower, upper)
    return _mask

def bool2mask(matrix, value=255):
    """
    Convert Boolean matrix to 8-bit unsigned integer mask image
    """
    if is_bool(matrix):
        result_int = matrix.astype(int)
        _mask = result_int * value
        return _mask.astype(np.uint8)
    else:
        raise ValueError("Input matrix must be of bool dtype.")

def inpaint_defect(image, mask, radius=10, flags=1):
    """
    Inpaint defect image
    :param image: defect image
    :param mask: Mask of defective parts
    :param radius: Inpaint radius
    :param flags: TELEA-1; NS-0
    """
    dst = cv2.inpaint(image, mask, radius, flags)
    return dst

def convert_mask(img_path, lower, upper, convert=(255, 255, 255)):
    """
    去除标签中的杂色,并对其进行颜色的转换

    :param img_path: 输入图像的文件路径
    :param lower: 颜色范围的下限值，作为一个包含(B, G, R)值的元组
    :param upper: 颜色范围的上限值，作为一个包含(B, G, R)值的元组
    :param convert: 需要转换为的颜色，作为一个包含(B, G, R)值的元组
    :return: 处理后的图像
    """
    image = cv2.imread(img_path)
    mask = up_low(image, lower, upper)
    image[mask > 0] = convert

    return image

def cvt8png(pngpath, bit_depth=False, target=(255, 255, 255), convert=(128, 0, 0)):
    """
    Voc: RGB png彩图转换
    :param bit_depth: 默认转为8位,需要使用out_png.save可以正确保存
    True:                      False:
        plt.imshow(img)             cv2.imshow("img",img)
        plt.axis('off')             cv2.waitKey(0)
        plt.show()
    :param pngpath: 为保证是按照cv2的方式读入, 所以传入路径即可
    :param target: 目标颜色, RGB方式
    :param convert: 转换颜色, 同RGB格式
    :return:
    """
    png = cv2.imread(pngpath)
    png = cv2.cvtColor(png, cv2.COLOR_BGR2RGB)
    h, w = png.shape[:2]

    mask = np.all(png == target, axis=-1)
    out_png = np.zeros((h, w, 3), dtype=np.uint8)
    out_png[mask] = convert
    out_png[~mask] = png[~mask]

    if bit_depth:
        out_png_pil = Image.fromarray(out_png)
        out_png_pil = out_png_pil.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        return out_png_pil
    else:
        out_png_cv = cv2.cvtColor(out_png, cv2.COLOR_RGB2BGR)
        return out_png_cv


def mask_foreground_move(image, H_pixels=None, V_pixels=None):
    if H_pixels is None:
        H_pixels = 0
    if V_pixels is None:
        V_pixels = 0
    h, w = image.shape[:2]
    black = np.zeros_like(image)

    start_row, end_row = max(0, V_pixels), min(h, h + V_pixels)
    start_col, end_col = max(0, H_pixels), min(w, w + H_pixels)

    fill_start_row, fill_end_row = max(0, -V_pixels), min(h, h - V_pixels)
    fill_start_col, fill_end_col = max(0, -H_pixels), min(w, w - H_pixels)

    black[start_row:end_row, start_col:end_col] = image[fill_start_row:fill_end_row, fill_start_col:fill_end_col]

    return black

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

G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                     0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
                     0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                     0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                     1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
                     0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)

G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
                      0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                      1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
                      0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)

def _generate_thin_luts():
    """generate LUTs for thinning algorithm (for reference)"""

    def nabe(n):
        return np.array([n >> i & 1 for i in range(0, 9)]).astype(bool)

    def G1(n):
        s = 0
        bits = nabe(n)
        for i in (0, 2, 4, 6):
            if not(bits[i]) and (bits[i + 1] or bits[(i + 2) % 8]):
                s += 1
        return s == 1

    g1_lut = np.array([G1(n) for n in range(256)])

    def G2(n):
        n1, n2 = 0, 0
        bits = nabe(n)
        for k in (1, 3, 5, 7):
            if bits[k] or bits[k - 1]:
                n1 += 1
            if bits[k] or bits[(k + 1) % 8]:
                n2 += 1
        return min(n1, n2) in [2, 3]

    g2_lut = np.array([G2(n) for n in range(256)])

    g12_lut = g1_lut & g2_lut

    def G3(n):
        bits = nabe(n)
        return not((bits[1] or bits[2] or not(bits[7])) and bits[0])

    def G3p(n):
        bits = nabe(n)
        return not((bits[5] or bits[6] or not(bits[3])) and bits[4])

    g3_lut = np.array([G3(n) for n in range(256)])
    g3p_lut = np.array([G3p(n) for n in range(256)])

    g123_lut = g12_lut & g3_lut
    g123p_lut = g12_lut & g3p_lut

    return bool2mask(g123_lut, value=1), bool2mask(g123p_lut, value=1)

if __name__=="__main__":
    image = cv2.imread(r"E:\PythonProject\pyzjrPyPi\pyzjr\test.png")
    mask, masked_image = create_rectmask(image, (50, 20), (180, 150))
    ulimg = up_low(image, (0, 0, 0), (255, 0, 255))
    approximg = approximate(image)
    ceilimg = ceilfloor(image)
    print(ceilimg.shape)
    # cv2.imshow("ss", mask)
    cv2.imshow("mask", ceilimg)
    cv2.waitKey(0)


















