"""
Copyright (c) 2025, Auorui.
All rights reserved.
"""
import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from urllib import request
from torchvision.utils import make_grid

_imshow = cv2.imshow  # copy to avoid recursion errors
from pyzjr.utils.randfun import randstring


def pil_to_ndarray(image):
    """Convert PIL images to OpenCV images"""
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if image.mode == 'L':
        image = np.array(image)
    else:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image


def ndarray_to_pil(image):
    """Convert an OpenCV image to a PIL image"""
    if image.ndim == 2:
        image = Image.fromarray(image, mode='L')
    else:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image


def tensor_to_ndarray(image, cols=8):
    """Convert a PyTorch Tensor (CHW or BCHW) to a displayable BGR Numpy array.
    If a batch tensor (B, C, H, W) is provided, it will be stitched into a grid."""
    image = image.detach().cpu()
    if image.dim() == 4:
        if image.size(0) == 1:
            image = image.squeeze(0)   # (C, H, W)
        else:
            image_list = [image[i] for i in range(image.size(0))]
            image = make_grid(image_list, nrow=int(cols))   # (C, H, W)

    if image.dim() == 3:
        if image.size(0) == 1:
            image = image.squeeze(0)
        else:
            image = image.permute(1, 2, 0)

    if image.dim() not in (2, 3):
        raise ValueError(f"The input tensor must be 2D (H,W), 3D (C,H,W), or 4D (B,C,H,W), got {image.dim()}D.")

    image = image.numpy()

    img_min = image.min()
    img_max = image.max()
    denom = img_max - img_min if img_max != img_min else 1.0
    image = (image - img_min) / denom
    # image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 3:
        image = image[:, :, ::-1].copy()
    return image


def ndarray_to_tensor(image, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    """Convert an OpenCV Numpy array (BGR) to a PyTorch Tensor (B, C, H, W)."""
    image_np = np.array(image)
    image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    image = image.to(device)
    return image


def read_image(filename, flags='bgr', transforms=None):
    """
    Read an image from a local path or a URL.
    Supports Chinese paths and multiple output formats.

    Args:
        filename (str or Path): The path or URL of the image.
        flags (str): Output format. Options: 'bgr', 'rgb', 'gray', 'unchanged', 'tensor'. Defaults to 'bgr'.
        transforms (list, optional): A list of callable functions to apply to the image sequentially.

    Returns:
        np.ndarray or torch.Tensor: The loaded image.
    """
    if transforms is not None and not all(callable(t) for t in transforms):
        raise ValueError("transforms must be a list of callable functions")
    if isinstance(filename, Path):
        filename = str(filename.resolve())
    if flags.lower() in ['gray', 'Gray', 'grayscale']:
        read_flags = cv2.IMREAD_GRAYSCALE
    elif flags.lower() in ['unchanged', 'any', 'alpha', 'Alpha']:
        read_flags = cv2.IMREAD_UNCHANGED
    else:
        read_flags = cv2.IMREAD_COLOR

    if isinstance(filename, str) and filename.startswith(('http://', 'https://')):
        try:
            res = request.urlopen(filename, timeout=5)
            img_array = np.asarray(bytearray(res.read()), dtype="uint8")
            img_np = cv2.imdecode(img_array, read_flags)
        except Exception as e:
            raise ValueError(f"Failed to load URL image: {e}")
    else:
        if not os.path.isfile(str(filename)):
            raise ValueError(f"The file {filename} does not exist.")
        img_np = cv2.imdecode(np.fromfile(filename, np.uint8), read_flags)

    if img_np is None:
        raise ValueError(f"Failed to decode image: {filename}")

    if transforms:
        for transform in transforms:
            img_np = transform(img_np)

    if flags.lower() == 'rgb':
        image = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    elif flags.lower() == 'tensor':
        image = ndarray_to_tensor(img_np)
    else:
        image = img_np.copy()
    return image


def write_image(filename: str, image, cols=8, params=None):
    """
    Write an image to a file, supporting Chinese paths and automatic type conversion.

    Args:
        filename (str): Target file path.
        image: Input image (Tensor, PIL, or Numpy array).
        cols (int): Number of grid columns if a batch Tensor is provided. Defaults to 8.
        params (list, optional): OpenCV compression parameters.

    Returns:
        bool: True if successfully saved, False otherwise.
    """
    if torch.is_tensor(image):
        image = tensor_to_ndarray(image, cols=cols)
    if isinstance(image, Image.Image):
        image = pil_to_ndarray(image)

    suffix = Path(filename).suffix
    if not suffix:
        raise ValueError(f"File extension is required, but got filename: {filename}")
    try:
        cv2.imencode(Path(filename).suffix, image, params)[1].tofile(filename)
        return True
    except Exception:
        return False


def show_image(image, winname=None, scale=1, cols=None, pad_color=(0, 0, 0)):
    """
    Display an image in a window. Supports Numpy, Tensor, PIL, and a list of Numpy arrays.

    Args:
        image: The input image(s).
        winname (str, optional): Window name. Randomly generated if None.
        scale (float): Display scaling factor. Defaults to 1.
        cols (int, optional): Number of grid columns for batch Tensor or image list.
    """
    if winname is None:
        winname = randstring(5)

    if isinstance(image, Image.Image):
        image = pil_to_ndarray(image)

    elif torch.is_tensor(image):
        if cols is None:
            cols = min(8, len(image))
        image = tensor_to_ndarray(image, cols=cols)

    elif isinstance(image, (list, tuple)) and all(isinstance(img, np.ndarray) for img in image):
        image = stacked_images(image, scale=scale, cols=cols, pad_color=pad_color)

    if isinstance(image, np.ndarray) and scale != 1.0:
        h, w = image.shape[:2]
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    safe_winname = winname.encode('gbk', errors='ignore').decode('gbk')
    _imshow(safe_winname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def stacked_images(imgList, scale=1.0, cols=None, pad_color=(0, 0, 0)):
    """
    Combine multiple images into a single grid layout for easy display.

    Args:
        imgList (list): A list of images (NumPy arrays or Tensors).
        cols (int, optional): Number of columns. If None, automatically calculates
                              a near-square grid. Defaults to None.
        scale (float): Scaling factor applied to each image. Defaults to 1.0.
        pad_color (tuple): RGB/BGR color for blank padding slots. Defaults to (0, 0, 0).

    Returns:
        np.ndarray: The combined grid image (BGR).
    """
    if not imgList:
        raise ValueError("imgList cannot be empty.")

    processed_imgs = []
    for img in imgList:
        if img.dtype != np.uint8:
            if img.max() <= 1.0 and img.min() >= 0.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        processed_imgs.append(img)

    total = len(processed_imgs)

    if cols is None:
        cols = int(np.ceil(np.sqrt(total)))
    else:
        cols = int(cols)

    rows = int(np.ceil(total / cols))

    if scale != 1.0:
        processed_imgs = [
            cv2.resize(img, (0, 0), None, scale, scale) for img in processed_imgs
        ]

    img_h, img_w = processed_imgs[0].shape[:2]

    canvas_w = cols * img_w
    canvas_h = rows * img_h
    canvas = np.full((canvas_h, canvas_w, 3), pad_color, dtype=np.uint8)

    for idx, img in enumerate(processed_imgs):
        r = idx // cols
        c = idx % cols

        if img.shape[:2] != (img_h, img_w):
            img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        y_start = r * img_h
        x_start = c * img_w
        canvas[y_start:y_start+img_h, x_start:x_start+img_w] = img

    return canvas



if __name__=="__main__":
    url_path = r'https://ts1.tc.mm.bing.net/th/id/OIP-C.owNHRpVQOhVjaii7UtbF9gHaE7?r=0&rs=1&pid=ImgDetMain&o=7&rm=3'
    image_path = r"/resources/images/aircraft.jpg"
    label_path = r"/resources/images/aircraft_mask.png"
    pil_image = Image.open(image_path).convert('RGB')

    image = read_image(image_path, flags='tensor')
    label = read_image(label_path, flags='tensor')

    from functools import partial

    flip = partial(cv2.flip, flipCode=0)

    transforms = [
        flip,
        lambda img: cv2.resize(img, (256, 256)),
    ]

    url_image = read_image(url_path, flags='rgb', transforms=transforms)
    print(url_image.shape)

    batch_images = torch.cat([image, label, image, label], dim=0)
    print(batch_images.shape)

    show_image(batch_images, scale=.5)

    # show_image([url_image, url_image, url_image, url_image], cols=4)

    # write_image("file.png", batch_images)
    print(image.shape)

