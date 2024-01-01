import torch
import numpy as np
import torchvision.transforms.functional as F
from .general import is_numpy
from .error import check_dtype

__all__=[    "hwc_to_chw",
             "to_bchw",
             "image_to_tensor",
             "imagelist_to_tensor",
             "tensor_to_image",
             "img2tensor",
             "label2tensor",
        ]

Tensor = torch.Tensor

def hwc_to_chw(img):
    """Transpose the input image from shape (H, W, C) to (C, H, W)."""
    if not is_numpy(img):
        raise TypeError('img should be NumPy array. Got {}.'.format(type(img)))
    if img.ndim not in (2, 3):
        raise TypeError("img dimension should be 2 or 3. Got {}.".format(img.ndim))
    if img.ndim == 2:
        return img
    return img.transpose(2, 0, 1).copy()


def to_bchw(tensor):
    if not isinstance(tensor, Tensor):
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

def image_to_tensor(image, keepdim=True):
    """Convert a numpy image to a PyTorch 4d tensor image.
        Example:
            img = np.ones((3, 3))
            image_to_tensor(img).shape
        torch.Size([1, 3, 3])

            img = np.ones((4, 4, 1))
            image_to_tensor(img).shape
        torch.Size([1, 4, 4])

            img = np.ones((4, 4, 3))
            image_to_tensor(img, keepdim=False).shape
        torch.Size([1, 3, 4, 4])
    """
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

def tensor_to_image(tensor: Tensor, keepdim: bool = False):
    """Converts a PyTorch tensor image to a numpy image.
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
    if not isinstance(tensor, Tensor):
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

def img2tensor(im, normalize=None):
    """NumPy图像数组转换为PyTorch张量"""
    tensor = torch.from_numpy(np.moveaxis(im / (255.0 if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor

def label2tensor(mask, num_classes, sigmoid):
    """标签或掩码图像转换为 PyTorch 张量"""
    if num_classes > 1:
        if not sigmoid:
            long_mask = np.zeros((mask.shape[:2]), dtype=np.int64)
            if len(mask.shape) == 3:
                for c in range(mask.shape[2]):
                    long_mask[mask[..., c] > 0] = c
            else:
                long_mask[mask > 127] = 1
                long_mask[mask == 0] = 0
            mask = long_mask
        else:
            mask = np.moveaxis(mask / (255.0 if mask.dtype == np.uint8 else 1), -1, 0).astype(np.float32)
    else:
        mask = np.expand_dims(mask / (255.0 if mask.dtype == np.uint8 else 1), 0).astype(np.float32)
    return torch.from_numpy(mask)

def SumExceptBatch(x, num_batch_dims=1):
    """
    求和“x”中除第一个“num_batch_dims”维度外的所有元素。
    case:
    x1 = torch.tensor([[1, 2], [3, 4]])
    result1 = SumExceptBatch(x1, num_batch_dims=1)
    >> tensor([3, 7])
    """
    check = check_dtype(num_batch_dims)
    if not check.is_nonnegative_int():
        raise TypeError("Number of batch dimensions must be a non-negative integer.")
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)