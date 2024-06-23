"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is applied to image reading and display.
"""
import cv2
import torch
import numpy as np
from urllib import request
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from pathlib import Path
from pyzjr.core.general import is_gray_image, is_pil, is_url, is_file, is_numpy, is_list
from pyzjr.visualize.color.colorspace import bgr2rgb
from pyzjr.augmentation.PIC import convert_cv_to_pil

IMREAD_GRAYSCALE = cv2.IMREAD_GRAYSCALE
IMREAD_COLOR = cv2.IMREAD_COLOR
RGB2BGR  = cv2.COLOR_RGB2BGR
BGR2RGB  = cv2.COLOR_BGR2RGB
BGR2GRAY = cv2.COLOR_BGR2GRAY
RGB2GRAY = cv2.COLOR_RGB2GRAY
GRAY2BGR = cv2.COLOR_GRAY2BGR
GRAY2RGB = cv2.COLOR_GRAY2RGB
BGR2HSV  = cv2.COLOR_BGR2HSV
RGB2HSV  = cv2.COLOR_RGB2HSV
HSV2BGR  = cv2.COLOR_HSV2BGR
HSV2RGB  = cv2.COLOR_HSV2RGB

def display(winname, imgArray, scale=1.):
    """Displays an image in the specified window."""
    _imshow = cv2.imshow  # copy to avoid recursion errors
    if is_numpy(imgArray):
        height, width = imgArray.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(imgArray, (new_width, new_height))
        _imshow(winname.encode('unicode_escape').decode(), image)
    elif is_list(imgArray):
        image = StackedImagesV1(scale, imgArray)
        _imshow(winname.encode('unicode_escape').decode(), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image

def imreader(filename, is_gray=False, flip=None, to_rgb=False):
    """
    :param filename: Path to the image file.
    :param is_gray: If True, read the image in grayscale.
    :param flip: 0: Flip vertically.
                1: Flip horizontally.
               -1: Flip both vertically and horizontally.
    :param to_rgb: Convert to RGB, default to False
    """
    if not is_file(filename):
        raise ValueError(f"imreader: The file {filename} does not exist.")
    if isinstance(filename, Path):
        filename = str(filename.resolve())
    flags = IMREAD_GRAYSCALE if is_gray else IMREAD_COLOR
    image = cv2.imdecode(np.fromfile(filename, np.uint8), flags)
    if flip is not None:
        assert flip in [-1, 0, 1], f"imreader: The 'flip' parameter must be -1, 0, or 1."
        image = cv2.flip(image, flip)
    if is_gray_image(image):
        image = cv2.cvtColor(image, GRAY2RGB) if to_rgb else image
    else:
        image = bgr2rgb(image) if to_rgb else image
    return image

def imwrite(filename: str, img: np.ndarray, params=None):
    """Write the image to a file."""
    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)
        return True
    except Exception:
        return False

def url2image(url):
    if is_url(url):
        res = request.urlopen(url, timeout=3)
    else:
        raise ValueError("url2image: The current input parameter does not conform to the URL format")
    try:
        image = np.asarray(bytearray(res.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    except:
        print('[url2imgcv2]: Load read - Image timeout!')
        image = []
    h, w, c = image.shape
    if c == 4:
        image = image[:, :, :3]
    return image

def improperties(image):
    if is_numpy(image):
        imshape = image.shape
        size = imshape[:2]
        dtype = image.dtype
        height, width = size

        if len(imshape) == 2:
            depth = image.itemsize * 8
        else:
            depth = image.itemsize * 8 * imshape[2]

        return {
            "shape": imshape,
            "size": size,
            "height": height,
            "width": width,
            "dtype": dtype,
            "depth": depth,
            "source": "NumPy"
        }
    elif is_pil(image):
        width, height = image.size
        mode = image.mode

        if mode == 'L':  # Grayscale
            depth = 8
        elif mode in ('RGB', 'RGBA'):
            depth = 8 * 3 if mode == 'RGB' else 8 * 4
        elif mode in ('I', 'F'):  # Integer or float (single channel)
            depth = image.getbands()[0].size * 8
        else:
            depth = "Unknown"  # Other modes are not handled here

        return {
            "shape": (height, width, len(image.getbands())),  # Simulate a NumPy shape
            "size": (width, height),
            "height": height,
            "width": width,
            "dtype": "PIL mode " + mode,  # PIL doesn't have a direct dtype equivalent
            "depth": depth,
            "source": "PIL"
        }
    else:
        return "Input is not a NumPy array or PIL Image."

def imshowplt(image, stacked_shape=None, title=None, cmap=None, save_path=None):
    """显示图像并添加标题"""
    import matplotlib
    matplotlib.use("TkAgg")
    if is_list(image):
        image = Stackedplt(image, stacked_shape)
    else:
        if is_pil(image):
            if image.mode == 'L' or (image.mode == 'LA' and image.getbands()[1] == 'L'):
                cmap = 'gray'
        elif is_gray_image(image):
            cmap = 'gray'
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def StackedImagesV1(scale, imgArray):
    """
    :param scale:图片的规模, 可使用小数, 1为原始图像
    :param imgArray: 图像组成的列表，来表示行列排列
    :return: 生成的图像可按输入列表的行列排序展示, 拼接展示图像
    """
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def StackedImagesV2(scale, imgList, cols):
    """
    将多张图像堆叠在一起并在单个窗口中显示
    :param scale: 图像的缩放比例,大于1表示放大,小于1表示缩小
    :param imgList: 要堆叠的图像列表
    :param cols: 每行显示的图像数量
    :return: 堆叠后的图像
    """
    totalImages = len(imgList)
    rows = totalImages // cols if totalImages // cols * cols == totalImages else totalImages // cols + 1
    blankImages = cols * rows - totalImages

    width = imgList[0].shape[1]
    height = imgList[0].shape[0]
    imgBlank = np.zeros((height, width, 3), np.uint8)
    imgList.extend([imgBlank] * blankImages)
    for i in range(cols * rows):
        imgList[i] = cv2.resize(imgList[i], (0, 0), None, scale, scale)
        if len(imgList[i].shape) == 2:
            imgList[i] = cv2.cvtColor(imgList[i], cv2.COLOR_GRAY2BGR)
    hor = [imgBlank] * rows
    for y in range(rows):
        line = []
        for x in range(cols):
            line.append(imgList[y * cols + x])
        hor[y] = np.hstack(line)
    ver = np.vstack(hor)
    return ver

def Stackedplt(images, shape):
    """
    按照指定形状堆叠多张图片
    :param images: 包含多张PIL Image对象的列表
    :param shape: 展示的图片的行列数,如(2, 3)
    :return: 堆叠后的PIL Image对象
    """
    rows, cols = shape
    new_images = []
    for im in images:
        if is_pil(im):
            new_images.append(im)
        elif is_numpy(im):
            new_images.append(convert_cv_to_pil(im))
    assert all(img.size == new_images[0].size for img in new_images), "All images must be the same size"
    width, height = new_images[0].size
    new_img_width = width * cols
    new_img_height = height * rows
    new_img = Image.new('RGB', (new_img_width, new_img_height), color=(255, 255, 255))
    x_offset = 0
    y_offset = 0
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < len(new_images):
                img = new_images[index]
                new_img.paste(img, (x_offset, y_offset))
                x_offset += width
        x_offset = 0
        y_offset += height

    return new_img

def Stackedtorch(imgs, num_rows, num_cols, titles=None, scale=None, camp=None):
    """
    堆叠显示Tensor图像或PIL图像
    :param imgs: 包含图像的列表,可以包含Tensor图像或PIL图像
    :param num_rows: 图像的行数
    :param num_cols: 图像的列数
    :param titles: 图像的标题列表,默认为None
    :param scale: 图像的缩放比例,默认为None
    :param camp: 'gray'
    :return:
    example:
        imgs = [image1, image2, image3, image4]
        titles = ['Image 1', 'Image 2', 'Image 3','Image 4']
        axes = Stackedtorch(imgs, 2, 2, titles=titles, scale=5)
        plt.savefig('my_plot.jpg')
        plt.show()
    """
    if scale:
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    else:
        _, axes = plt.subplots(num_rows, num_cols)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy(), cmap=camp)
        else:
            # PIL Image
            ax.imshow(img, cmap=camp)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])

    return axes

def plot_line(x, y, xlabel, ylabel, title):
    """
    绘制折线图
    :param x: x轴的数据列表
    :param y: y轴的数据列表
    :param xlabel: x轴的标签
    :param ylabel: y轴的标签
    :param title: 图形的标题
    :return:
    """
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def bar_chart(x, y, xlabel, ylabel, title):
    """绘制柱状图"""
    plt.bar(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def scatter_plot(x, y, xlabel, ylabel, title):
    """绘制散点图"""
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


if __name__=="__main__":
    from PIL import Image
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    xlabel = 'X'
    ylabel = 'Y'
    title = '折线图示例'
    # plot_line(x, y, xlabel, ylabel, title)
    # bar_chart(x, y, xlabel, ylabel, title)
    # scatter_plot(x, y, xlabel, ylabel, title)

    image_path = r'D:\PythonProject\pyzjrPyPi\pyzjr\augmentation\test.png'
    # image = imreader(image_path)
    # gray_image = imreader(image_path, is_gray=True)
    # print(improperties(image), '\n', improperties(gray_image))
    # display("image-test", [[image, image], [gray_image, gray_image]], 1)
    # imshowplt([gray_image, image, gray_image, gray_image], (2,2))
    # imwrite("ss.png", image)

    image_pil = Image.open(image_path)
    print(improperties(image_pil))
    # imagess = Stackedtorch([image_pil, image_pil], 1, 2)
    import matplotlib
    matplotlib.use("TkAgg")
    axes = Stackedtorch([image_pil, image_pil], 1, 2, scale=5, titles=["dog1", "dog2"])
    plt.show()