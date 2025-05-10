"""
Copyright (c) 2022, Auorui.
All rights reserved.
"""
import cv2
import torch
import numpy as np
_imshow = cv2.imshow  # copy to avoid recursion errors
from pathlib import Path
from pyzjr.utils.check import is_file, is_numpy, is_list, is_pil, is_tensor
from pyzjr.augmentation.augments import resizepad
from pyzjr.utils.randfun import randstring

def read_gray(filename):
    if not is_file(filename):
        raise ValueError(f"The file {filename} does not exist.")
    if isinstance(filename, Path):
        filename = str(filename.resolve())
    return cv2.imdecode(np.fromfile(filename, np.uint8), cv2.IMREAD_GRAYSCALE)

def read_bgr(filename):
    if not is_file(filename):
        raise ValueError(f"The file {filename} does not exist.")
    if isinstance(filename, Path):
        filename = str(filename.resolve())
    return cv2.imdecode(np.fromfile(filename, np.uint8), cv2.IMREAD_COLOR)

def read_rgb(filename):
    if not is_file(filename):
        raise ValueError(f"The file {filename} does not exist.")
    if isinstance(filename, Path):
        filename = str(filename.resolve())
    return cv2.imdecode(np.fromfile(filename, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]

def read_tensor(filename, target_shape, device=None, pad_color=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rgb_image = read_rgb(filename)
    if pad_color:
        image = resizepad(rgb_image, target_shape, pad_color)
    else:
        image = cv2.resize(rgb_image, target_shape)
    image = image.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image_tensor.to(device)

def read_image(filename, flags='rgb', target_shape=None, pad_color=None):
    if not is_file(filename):
        raise ValueError(f"The file {filename} does not exist.")
    if isinstance(filename, Path):
        filename = str(filename.resolve())
    image = None
    if flags in ['bgr', 'BGR']:
        image = cv2.imdecode(np.fromfile(filename, np.uint8), cv2.IMREAD_COLOR)
    elif flags in ['rgb', 'RGB', 'torch', 'tensor', 'Tensor']:
        image = cv2.imdecode(np.fromfile(filename, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
    elif flags in ['alpha', 'Alpha']:
        image = cv2.imdecode(np.fromfile(filename, np.uint8), cv2.IMREAD_UNCHANGED)
    elif flags in ['gray', 'Gray']:
        image = cv2.imdecode(np.fromfile(filename, np.uint8), cv2.IMREAD_GRAYSCALE)
    if target_shape:
        if pad_color:
            image = resizepad(image, target_shape, pad_color)
        else:
            image = cv2.resize(image, target_shape)

    if flags in ['torch', 'tensor', 'Tensor']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image = image.to(device)
    return image

def imwrite(filename: str, image, params=None):
    """Write the image to a file."""
    if is_tensor(image):
        image = image.detach().cpu().squeeze(0)
        image = image.permute(1, 2, 0).numpy()
        image = (image - image.min()) / (image.max() - image.min())
        image = (image[:, :, ::-1] * 255).astype(np.uint8)
    try:
        cv2.imencode(Path(filename).suffix, image, params)[1].tofile(filename)
        return True
    except Exception:
        return False

def display(image, winname=None, scale=1.):
    """Displays an image in the specified window."""
    if winname is None:
        winname = randstring(5)
    if is_numpy(image):
        height, width = image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    elif is_list(image):
        image = StackedImagesV1(scale, image)
    _imshow(winname.encode('unicode_escape').decode(), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
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

if __name__=="__main__":
    image_path = r"E:\PythonProject\pyzjrPyPi\pyzjr\test.png"
    image = read_image(image_path, flags='rgb')
    display(image)
    print(image.shape)