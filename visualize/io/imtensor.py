import torch
import numpy as np
from pyzjr.utils.check import is_numpy

def image_to_bchw(image_data):
    """
    将图像加载为可输入网络的形状 b c h w, 且 b = 1
    """
    image_np = np.array(image_data).astype(np.uint8) / 255
    image_bchw = np.expand_dims(np.transpose(image_np[:, :, ::-1], (2, 0, 1)), 0)
    return image_bchw

def bchw_to_image(image_bchw):
    """
    将网络输出转为图像类型, 且 b = 1
    """
    image = image_bchw.detach().cpu().squeeze(0)
    image = image.permute(1, 2, 0).numpy()
    image = (image - image.min()) / (image.max() - image.min())
    image = (image[:, :, ::-1] * 255).astype(np.uint8)
    return image

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