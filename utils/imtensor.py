import cv2
import torch
import numpy as np
from typing import Iterable
import torch.nn.functional as F

from pyzjr.utils.check import is_pil, is_tensor, is_numpy, is_list_or_tuple
from pyzjr.nn.strategy import preprocess_input
from pyzjr.augmentation.mask_ops import convert_np


def to_numpy(x, dtype=None):
    if is_pil(x):
        return np.array(x, dtype=dtype)
    elif is_tensor(x):
        numpy_array = x.cpu().numpy()
        if dtype is not None:
            numpy_array = numpy_array.astype(dtype)
        return numpy_array
    elif is_numpy(x):
        if dtype is not None:
            return x.astype(dtype)
        return x
    elif isinstance(x, (Iterable, int, float)):
        return np.array(x, dtype=dtype)
    elif is_list_or_tuple(x):
        return np.array(x, dtype=dtype)
    else:
        raise ValueError("Unsupported type")

def to_tensor(x, dtype=None):
    if is_tensor(x):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if is_numpy(x):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if is_list_or_tuple(x):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    else:
        raise ValueError("Unsupported type")

def hwc2chw(img):
    """
    Conversion from 'HWC' to 'CHW' format.
    Example:
        hwc_image_numpy = np.random.rand(256, 256, 3)
        chw_image_numpy = hwc2chw(hwc_image_numpy)
        hwc_image_tensor = torch.rand(256, 256, 3)
        chw_image_tensor = hwc2chw(hwc_image_tensor)
    """
    if len(img.shape) == 3:
        if is_numpy(img):
            chw = np.transpose(img, axes=[2, 0, 1])
            return chw
        elif is_tensor(img):
            chw = img.permute(2, 0, 1).contiguous()
            return chw
        else:
            raise TypeError("The input data should be a NumPy array or "
                            "PyTorch tensor, but the provided type is: {}".format(type(img)))
    else:
        raise ValueError("The input data should be three-dimensional (height x width x channel), but the "
                         "provided number of dimensions is:{}".format(len(img.shape)))

def chw2hwc(img):
    """Conversion from 'CHW' to 'HWC' format."""
    if len(img.shape) == 3:
        if is_numpy(img):
            hwc = np.transpose(img, axes=[1, 2, 0])
            return hwc
        elif is_tensor(img):
            hwc = img.permute(1, 2, 0).contiguous()
            return hwc
        else:
            raise TypeError("The input data should be a NumPy array or "
                            "PyTorch tensor, but the provided type is: {}".format(type(img)))
    else:
        raise ValueError ("The input data should be three-dimensional (channel x height x width), but the "
                          "provided number of dimensions is: {}".format(len(img.shape)))

def to_bchw(tensor):
    """
    Convert to 'bchw' format
    Example:
        image_tensor = torch.rand(256, 256)
        bchw_image_tensor = to_bchw(image_tensor)
        print("Original shape:", image_tensor.shape)
        print("Converted shape:", bchw_image_tensor.shape)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(tensor)}")

    if len(tensor.shape) < 2:
        raise ValueError(f"Input size must be a two, three or four dimensional tensor. Got {tensor.shape}")

    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)

    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)

    if len(tensor.shape) > 4:
        tensor = tensor.view(-1, tensor.shape[-3], tensor.shape[-2], tensor.shape[-1])

    return tensor

def image_to_bchw(image_data):
    """
    将图像加载为可输入网络的形状 b c h w, 且 b = 1
    """
    image_np = preprocess_input(convert_np(image_data))
    image_bchw = np.expand_dims(np.transpose(image_np, (2, 0, 1)), 0)
    return image_bchw

def bchw_to_image(image_bchw):
    """
    将网络输出转为图像类型, 且 b = 1
    """
    image_chw = image_bchw[0]  # chw
    image_hwc = image_chw.permute(1, 2, 0)
    image_np = F.softmax(image_hwc, dim=-1).cpu().numpy()
    image_np = np.clip(image_np * 255, 0, 255)
    image_np = image_np.astype(np.uint8)
    return image_np

def image_to_tensor(image, keepdim=True):
    """
    Convert numpy images to PyTorch 4d tensor images
    'keepdim' indicates whether to maintain the current dimension, otherwise it will be changed to type 4d
    Example:
        img = np.ones((3, 3))
        image_to_tensor(img).shape
    [1, 3, 3]
        img = np.ones((4, 4, 1))
        image_to_tensor(img).shape
    [1, 4, 4]
        img = np.ones((4, 4, 3))
        image_to_tensor(img, keepdim=False).shape
    [1, 3, 4, 4]
    """
    if is_numpy(image):
        if len(image.shape) > 4 or len(image.shape) < 2:
            raise ValueError("Input size must be a two, three or four dimensional array")
    input_shape = image.shape
    tensor = torch.from_numpy(image)

    if len(input_shape) == 2:
        # (H, W) -> (1, H, W)
        tensor = tensor.unsqueeze(0)
    elif len(input_shape) == 3:
        # (H, W, C) -> (C, H, W)
        tensor = tensor.permute(2, 0, 1)
    elif len(input_shape) == 4:
        # (B, H, W, C) -> (B, C, H, W)
        tensor = tensor.permute(0, 3, 1, 2)
        keepdim = True  # no need to unsqueeze
    else:
        raise ValueError(f"Cannot process image with shape {input_shape}")

    return tensor if keepdim else tensor.unsqueeze(0)

def tensor_to_image(tensor, keepdim = False):
    """Convert PyTorch tensor image to numpy image
    Returns:
        image of the form :math:`(H, W)`, :math:`(H, W, C)` or :math:`(B, H, W, C)`.
    Example:
        img = torch.ones(1, 3, 3)
        tensor_to_image(img).shape
    (3, 3)
        img = torch.ones(3, 4, 4)
        tensor_to_image(img).shape
    (4, 4, 3)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(tensor)}")

    if len(tensor.shape) > 4 or len(tensor.shape) < 2:
        raise ValueError("Input size must be a two, three or four dimensional tensor")

    input_shape = tensor.shape
    image = tensor.cpu().detach().numpy()

    if len(input_shape) == 2:
        # (H, W) -> (H, W)
        pass
    elif len(input_shape) == 3:
        # (C, H, W) -> (H, W, C)
        if input_shape[0] == 1:
            # Grayscale for proper plt.imshow needs to be (H,W)
            image = image.squeeze()
        else:
            image = image.transpose(1, 2, 0)
    elif len(input_shape) == 4:
        # (B, C, H, W) -> (B, H, W, C)
        image = image.transpose(0, 2, 3, 1)
        if input_shape[0] == 1 and not keepdim:
            image = image.squeeze(0)
        if input_shape[1] == 1:
            image = image.squeeze(-1)
    else:
        raise ValueError(f"Cannot process tensor with shape {input_shape}")

    return image

def imagelist_to_tensor(imagelist):
    """Converts a list of numpy images to a PyTorch 4d tensor image.
    Args:
        images: list of images, each of the form :math:`(H, W, C)`.
        Image shapes must be consistent
    Returns:
        tensor of the form :math:`(B, C, H, W)`.
    Example:
        imgs = [np.ones((4, 4, 1)),
                np.zeros((4, 4, 1))]
        image_list_to_tensor(imgs).shape
    torch.Size([2, 1, 4, 4])
    """
    if len(imagelist[0].shape) != 3:
        raise ValueError("Input images must be three dimensional arrays")
    list_of_tensors = []
    for image in imagelist:
        list_of_tensors.append(image_to_tensor(image))
    return torch.stack(list_of_tensors)

def img2tensor(bgr, to_tensor=True):
    img = convert_np(bgr)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    chwrgb_img = np.moveaxis(rgb_img, -1, 0).astype(np.float32)
    return torch.from_numpy(chwrgb_img).type(torch.FloatTensor) if to_tensor else chwrgb_img

def label2tensor(mask, num_classes, sigmoid, totensor=False):
    """标签或掩码图像转换为 PyTorch 张量"""
    mask = np.array(mask)
    if num_classes > 2:
        if not sigmoid:
            long_mask = np.zeros((mask.shape[:2]), dtype=np.int64)
            if len(mask.shape) == 3:
                for c in range(mask.shape[2]):
                    long_mask[mask[..., c] > 0] = c
            else:
                long_mask[mask >= 127] = 1
                long_mask[mask == 0] = 0
            mask = long_mask
        else:
            mask = np.moveaxis(mask, -1, 0).astype(np.float32)
    else:
        mask[mask >= 127] = 1
        mask[mask == 0] = 0
    return torch.from_numpy(mask).long() if totensor else mask

if __name__=="__main__":
    hwc_image_numpy = np.random.rand(256, 256, 3)
    chw_image_numpy = hwc2chw(hwc_image_numpy)
    print("Original HWC shape:", hwc_image_numpy.shape)
    print("Converted CHW shape:", chw_image_numpy.shape)
    hwc_image_tensor = torch.rand(256, 256, 3)
    chw_image_tensor = hwc2chw(hwc_image_tensor)
    print("Original HWC shape:", hwc_image_tensor.shape)
    print("Converted CHW shape:", chw_image_tensor.shape)