import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from typing import Iterable

def pil_to_ndarray(pil_image):
    """Convert PIL images to OpenCV images"""
    if pil_image.mode == 'L':
        open_cv_image = np.array(pil_image)
    else:
        open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return open_cv_image

def ndarray_to_pil(cv_image):
    """Convert an OpenCV image to a PIL image"""
    if cv_image.ndim == 2:
        pil_image = Image.fromarray(cv_image, mode='L')
    else:
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    return pil_image

def to_numpy(x, dtype=None):
    if isinstance(x, Image.Image):
        return np.array(x, dtype=dtype)
    elif isinstance(x, torch.Tensor):
        numpy_array = x.cpu().numpy()
        if dtype is not None:
            numpy_array = numpy_array.astype(dtype)
        return numpy_array
    elif isinstance(x, np.ndarray):
        if dtype is not None:
            return x.astype(dtype)
        return x
    elif isinstance(x, (Iterable, int, float)):
        return np.array(x, dtype=dtype)
    elif isinstance(x, (list, tuple)):
        return np.array(x, dtype=dtype)
    else:
        raise ValueError("Unsupported type")

def to_tensor(x, dtype=None):
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    else:
        raise ValueError("Unsupported type")


def hwc2chw(x):
    """
    Conversion from 'HWC' to 'CHW' format.
    Example:
        hwc_image_numpy = np.random.rand(256, 256, 3)
        chw_image_numpy = hwc2chw(hwc_image_numpy)
        hwc_image_tensor = torch.rand(256, 256, 3)
        chw_image_tensor = hwc2chw(hwc_image_tensor)
    """
    if isinstance(x, Image.Image):
        x = np.array(x)
    if len(x.shape) == 3:
        if isinstance(x, np.ndarray):
            chw = np.transpose(x, axes=[2, 0, 1])
            return chw
        elif isinstance(x, torch.Tensor):
            chw = x.permute(2, 0, 1).contiguous()
            return chw
        else:
            raise TypeError("The input data should be a NumPy array or "
                            "PyTorch tensor, but the provided type is: {}".format(type(x)))
    else:
        raise ValueError("The input data should be three-dimensional (height x width x channel), but the "
                         "provided number of dimensions is:{}".format(len(x.shape)))

def chw2hwc(x):
    """Conversion from 'CHW' to 'HWC' format."""
    if isinstance(x, Image.Image):
        x = np.array(x)
    if len(x.shape) == 3:
        if isinstance(x, np.ndarray):
            hwc = np.transpose(x, axes=[1, 2, 0])
            return hwc
        elif isinstance(x, torch.Tensor):
            hwc = x.permute(1, 2, 0).contiguous()
            return hwc
        else:
            raise TypeError("The input data should be a NumPy array or "
                            "PyTorch tensor, but the provided type is: {}".format(type(x)))
    else:
        raise ValueError ("The input data should be three-dimensional (channel x height x width), but the "
                          "provided number of dimensions is: {}".format(len(x.shape)))

def to_bchw(x):
    """Convert to 'bchw' format"""
    if len(x.shape) < 2:
        raise ValueError(f"Input size must be a two, three or four dimensional tensor. Got {x.shape}")

    if len(x.shape) == 2:
        if isinstance(x, torch.Tensor):
            x = x.unsqueeze(0)
        elif isinstance(x, np.ndarray):
            x = np.expand_dims(x, axis=0)

    if len(x.shape) == 3:
        if isinstance(x, torch.Tensor):
            x = x.unsqueeze(0)
        elif isinstance(x, np.ndarray):
            x = np.expand_dims(x, axis=0)

    if len(x.shape) > 4:
        if isinstance(x, torch.Tensor):
            x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        elif isinstance(x, np.ndarray):
            x = x.reshape((-1, x.shape[-3], x.shape[-2], x.shape[-1]))
    return x

def bgr_image_to_bchw_tensor(image_data, to_tensor=True, device="cuda:0" if torch.cuda.is_available() else "cpu"):
    """Load the image into the shape suitable for input to a network as b c h w, where b equals 1."""
    image_np = np.array(image_data).astype(np.uint8) / 255
    image_bchw = np.expand_dims(np.transpose(image_np[:, :, ::-1], (2, 0, 1)), 0)
    if to_tensor:
        image_tensor = torch.from_numpy(image_bchw).float()
        return image_tensor.to(device)
    else:
        return image_bchw

def bchw_tensor_to_bgr_image(image_bchw, nrow=8):
    """Convert network output to image type with b = 1"""
    image = image_bchw.detach().cpu()
    if image.dim() == 4:
        if image.size(0) == 1:
            image = image.squeeze(0)
        else:
            image = bchw_tensor_to_list(image)
            image = make_grid(image, nrow=nrow)
    image = image.permute(1, 2, 0).numpy()
    image = (image - image.min()) / (image.max() - image.min())
    image = (image[:, :, ::-1] * 255).astype(np.uint8)
    return image

def bchw_tensor_to_list(tensor: torch.Tensor):
    """Convert tensors in BCHW format to CHW lists"""
    if tensor.dim() != 4:
        raise ValueError(f"The input tensor must be 4-dimensional (B, C, H, W), but the resulting tensor is {tensor. dim()} dimensional")
    return [tensor[i] for i in range(tensor.size(0))]

def bgr_image_list_to_tensor(imagelist):
    """Converts a list of numpy images to a PyTorch 4d tensor image."""
    if len(imagelist[0].shape) != 3:
        raise ValueError("Input images must be three dimensional arrays")
    list_of_tensors = []
    for image in imagelist:
        list_of_tensors.append(bgr_image_to_bchw_tensor(image, to_tensor=True))
    return torch.stack(list_of_tensors)


