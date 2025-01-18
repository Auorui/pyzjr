"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is applied to image reading and display.
"""
import os
import cv2
import sys
import imageio
import numpy as np
from PIL import Image
from pathlib import Path
from urllib import request
from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from pyzjr.utils.check import (is_file, is_gray_image, is_numpy, is_list, is_url, is_pil,
                               is_tensor)
from pyzjr.visualize.colorspace import bgr2rgb
from pyzjr.utils.converter import cv2pil

def imreader(filename, is_gray=False, flip=None, to_rgb=False):
    """
    :param filename: Path to the image file.
    :param is_gray: If True, read the image in grayscale.
    :param flip: 0: Flip vertically.
                1: Flip horizontally.
               -1: Flip both vertically and horizontally.
    :param to_rgb: Convert to RGB, default to False

    Examples:
    ```
        image_path = r'dog.png'
        gray_image = imreader(image_path, is_gray=True, to_rgb=True)
        display("ss", gray_image)
        imwriter("ss.png", gray_image)
    ```
    """
    if not is_file(filename):
        raise ValueError(f"The file {filename} does not exist.")
    if isinstance(filename, Path):
        filename = str(filename.resolve())
    flags = cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_COLOR
    image = cv2.imdecode(np.fromfile(filename, np.uint8), flags)
    if flip is not None:
        assert flip in [-1, 0, 1], f"The 'flip' parameter must be -1, 0, or 1."
        image = cv2.flip(image, flip)
    if is_gray_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if to_rgb else image
    else:
        image = bgr2rgb(image) if to_rgb else image
    return image

def imwriter(filename: str, img: np.ndarray, params=None):
    """Write the image to a file."""
    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)
        return True
    except Exception:
        return False

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

def url2image(url):
    """Same usage as cv2.imread()"""
    if is_url(url):
        res = request.urlopen(url, timeout=3)
    else:
        raise ValueError("The current input parameter does not conform to the URL format")
    try:
        image = np.asarray(bytearray(res.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    except:
        print('Load read - Image timeout!')
        image = []
    h, w, c = image.shape
    if c == 4:
        image = image[:, :, :3]
    return image

def imattributes(image):
    """Retrieve image attributes"""
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
    """
    Display image and add title
    Examples:
        color_image = Image.open("dog.png")
        imshowplt([color_image,color_image], stacked_shape=(2,1), title='PIL Color Image', cmap=None)
    """
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

class VideoCap():
    """
    Customized Python video reading class
    Examples:
    ```
        Vcap = VideoCap(mode=0)
        while True:
            img = Vcap.read()
            Vcap.show("ss", img)
    ```
    """
    def __init__(self, mode=0, width=640, height=480, light=150):
        self.cap = cv2.VideoCapture(mode)
        self.cap.set(3, width)
        self.cap.set(4, height)
        self.cap.set(10, light)
        self.start_number = 0

    def read(self, flip=None):
        """
        :param flip: -1: Horizontal and vertical directions,
                      0: along the y-axis, vertical,
                      1: along the x-axis, horizontal
        """
        _, img = self.cap.read()
        if flip is not None:
            assert flip in [-1, 0, 1], f"VideoCap: The 'flip' parameter must be -1, 0, or 1."
            img = cv2.flip(img, flip)
        return img

    def free(self):
        """
        Release camera
        """
        self.cap.release()
        cv2.destroyAllWindows()

    def show(self,
             winname: str,
             src,
             base_name: str = './result.png',
             end_k=27,
             save_k=ord('s'),
             delay_t=1,
             extend_num=3
             ):
        """
        Window display. Press 's' to save, 'Esc' to end
        """
        image_path, ext = os.path.splitext(base_name)
        os.makedirs(os.path.dirname(base_name), exist_ok=True)
        if src is not None:
            cv2.imshow(winname, src)
            k = cv2.waitKey(delay_t) & 0xFF
            if k == end_k:
                self.free()
                sys.exit(0)
            elif k == save_k:
                self.start_number += 1
                file_number = str(self.start_number).zfill(extend_num)
                file_path = f"{image_path}_{file_number}{ext}"
                print(f"{self.start_number}  Image saved to {file_path}")
                cv2.imwrite(file_path, src)

def StackedImagesV1(scale, imgArray):
    """
    Display Images According to List Structure

    :param scale: The scale of the images, where 1 represents the original size.
    :param imgArray: A list of images representing the arrangement in rows and columns.
    :return: A generated image that displays the images in the order specified by the input list, arranged in a grid.
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
    Combine multiple images into a single display within a single window
    :param scale: The scaling factor for the images, where a value greater than 1 indicates enlargement and a value less than 1 indicates reduction.
    :param imgList: A list of images to be combined.
    :param cols: The number of images to display per row.
    :return: The combined image.
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
    Stack multiple images according to the specified shape
    :param images: A list containing multiple PIL Image objects
    :param shape: The number of rows and columns for displaying the images, e.g., (2, 3)
    :return: A stacked PIL Image object
    """
    rows, cols = shape
    new_images = []
    for im in images:
        if is_pil(im):
            new_images.append(im)
        elif is_numpy(im):
            new_images.append(cv2pil(im))
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
    Display images in a grid format, which can include either Tensor images or PIL images.
    :param imgs: A list containing the images. The list can include both Tensor images and PIL images.
    :param num_rows: The number of rows in the grid.
    :param num_cols: The number of columns in the grid.
    :param titles: A list of titles for the images. Defaults to None.
    :param scale: The scaling factor for the images. Defaults to None, which means no scaling is applied.
    :param camp: 'gray'
    :return:
    Examples:
    ```
        imgs = [image1, image2, image3, image4]
        titles = ['Image 1', 'Image 2', 'Image 3','Image 4']
        axes = Stackedtorch(imgs, 2, 2, titles=titles, scale=5)
        plt.savefig('my_plot.jpg')
        plt.show()
    ```
    """
    if scale:
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    else:
        _, axes = plt.subplots(num_rows, num_cols)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if is_tensor(img):
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

def Mp4toGif(mp4, name='result.gif', fps=10, start=None, end=None):
    """Convert MP4 files to GIF animations"""
    cap = cv2.VideoCapture(mp4)
    all_images = []
    frame_count = 0
    while True:
        ret, img = cap.read()
        if ret is False:
            break
        if start is not None and frame_count < start:
            frame_count += 1
            continue
        if end is not None and frame_count >= end:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        all_images.append(img)
        frame_count += 1

    duration = int(1000 / fps)  # 将帧率转换为每帧之间的延迟时间（毫秒）
    imageio.mimsave(name, all_images, duration=duration)
    print("Conversion completed！")