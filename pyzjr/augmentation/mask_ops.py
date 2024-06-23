import cv2
import numpy as np
from PIL import Image
from pyzjr.core.general import is_numpy, is_bool
from skimage.morphology import medial_axis
from scipy.ndimage import distance_transform_edt

def convert_np(image, dtype=np.uint8):
    """Convert an image to a NumPy array with the specified dtype."""
    return image if is_numpy(image) else np.array(image).astype(dtype)

def clip(image, maxval, dtype=np.uint8):
    """
    Truncate the pixel values of the image to the specified range and perform data type conversion
    """
    np_image = convert_np(image)
    return np.clip(np_image, 0, maxval).astype(dtype)

def unique(image):
    """Returns a list of unique pixel values in the input image"""
    np_image = convert_np(image)
    return np.unique(np_image)

def RectMask(image, up_left_coord=None, lower_right_coord=None, bboxes=None):
    """
    Create a rectangular mask
    :param image: Original image.
    :param up_left_coord: [x1, y1]
    :param lower_right_coord: [x2, y2]
    :param bboxes: [x1, y1, x2, y2]
    """
    image = convert_np(image)
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

def BinaryImg(image, min_value=127, max_value=255):
    """
    Convert image to binary image
    :param image: Original image. --- (BGR format)
    :param min_value: Minimum threshold, default to 127
    :param max_value: Maximum threshold, default to 225
    """
    np_image = convert_np(image)
    gray_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, min_value, max_value, cv2.THRESH_BINARY)
    return binary_image

def up_low(image, lower, upper, dtype=np.uint8):
    """
    Create a binarized mask based on the given color range (lower and upper limits)
    :param lower: Lower limit of color range, (B, G, R)
    :param upper: Upper limit of color range, (B, G, R)
    """
    np_image = convert_np(image)
    lower = np.array(lower, dtype=dtype)
    upper = np.array(upper, dtype=dtype)
    _mask = cv2.inRange(np_image, lower, upper)
    return _mask

def approximate(image, std=127.5, dtype=np.uint8):
    """
    Convert a single channel image into a binary image.
    """
    image[image > std] = 255
    image[image < std] = 0
    image = image.astype(dtype)
    return image

def ceilfloor(image, dtype=np.uint8):
    """
    The pixel value of the input image is limited between the maximum value of 255 and the minimum value of 0
    """
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype(dtype)
    return image

def bool2mask(matrix):
    """
    Convert Boolean matrix to 8-bit unsigned integer mask image
    """
    if is_bool(matrix):
        result_int = matrix.astype(int)
        _mask = result_int * 255
        return _mask.astype(np.uint8)
    else:
        raise ValueError("Input matrix must be of bool dtype.")

def cv_distance(image):
    """Using distance transformation functions in OpenCV"""
    dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 3)
    return dist_transform

def chamfer(image, weight=None):
    """Chamfer distance transformation"""
    if weight is None:
        weights = np.array([[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]], dtype=np.uint8)
    else:
        weights = weight
    dist_transform = cv2.filter2D(image, cv2.CV_32F, weights)
    return dist_transform

def fast_marching(image):
    """Fast-Marching Distance Transform"""
    _, medial_axis_image = medial_axis(image, return_distance=True)   # 使用medial_axis函数计算中轴线
    dist_transform = distance_transform_edt(medial_axis_image)
    return dist_transform

def addnoisy(image, n=10000):
    """
    Add salt and pepper treatment
    """
    result = image.copy()
    w, h = image.shape[:2]
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
    基于大气散射模型对输入的图像添加雾霾效果的高效实现。

    Args:
        image (numpy.ndarray): 输入的图像,范围在0-255.
        beta (float, optional): 控制雾霾效果的参数. beta值越大, 雾霾效果越明显. 默认为0.05.
        brightness (float, optional): 雾霾的亮度值. 该值越大, 图像整体亮度越高. 默认为0.5.

    Returns:
        numpy.ndarray: 添加雾霾效果后的图像，数据类型为uint8，范围在0-255。
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

def medial_axis_mask(image):
    """
    The input must be a binary graph to obtain the axis image within it
    """
    result = medial_axis(image)
    return bool2mask(result)

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
        out_png_pil = out_png_pil.convert("P", palette=Image.ADAPTIVE, colors=256)
        return out_png_pil
    else:
        out_png_cv = cv2.cvtColor(out_png, cv2.COLOR_RGB2BGR)
        return out_png_cv

def cvtMask(pngpath, lower ,upper, convert=(255, 255, 255)):
    """
    去除标签中的杂色,并对其进行颜色的转换

    :param pngpath: 输入图像的文件路径
    :param lower: 颜色范围的下限值，作为一个包含(B, G, R)值的元组
    :param upper: 颜色范围的上限值，作为一个包含(B, G, R)值的元组
    :param convert: 需要转换为的颜色，作为一个包含(B, G, R)值的元组
    :return: 处理后的图像
    """
    image = cv2.imread(pngpath)
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)

    mask = cv2.inRange(image, lower, upper)
    image[mask > 0] = convert

    return image

def imageContentMove(image, H_pixels=None, V_pixels=None):
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

def count_zero(thresh):
    """计算矩阵0值"""
    zero_count = np.sum(thresh == 0)
    return zero_count

def count_white(thresh):
    """计算矩阵255值"""
    white_count = np.sum(thresh == 255)
    return white_count

def count_nonzero(thresh):
    """计算矩阵非0值"""
    nonzero_pixels = np.count_nonzero(thresh)
    return nonzero_pixels

if __name__=="__main__":
    image = np.zeros((10, 10), dtype=np.uint8)
    image[2:8, 2:8] = 255
    print(image)

    dist_transform = cv_distance(image)
    chamfer_dist_transform = chamfer(image)
    fast_marching_dist_transform = fast_marching(image)
    # 距离变换结果
    print("Distance Transform:")
    print(dist_transform)
    print("\nChamfer Distance Transform:")
    print(chamfer_dist_transform)
    print("\nFast-Marching Distance Transform:")
    print(fast_marching_dist_transform)

    mask_img_path = r'D:\PythonProject\pyzjrPyPi\models_img\1604.png'
    mask_img = cv2.imread(mask_img_path)
    img = clip(mask_img, 255)
    # img = up_low(img, (0, 0, 100), (255, 255, 255))
    # img = approximate(img, 130)
    img = BinaryImg(img)
    # img = ceilfloor(img)
    # _, img = RectMask(img, (15, 27), (388, 300))
    result = medial_axis_mask(img)
    print(unique(result))
    # print(result.shape)
    cv2.imshow("test mask-function", result)
    cv2.waitKey(0)

