import torch
import numpy as np
from pyzjr.core.general import is_numpy, is_tensor, is_nonnegative_int

def SumExceptBatch(x, num_batch_dims=1):
    """
    求和“x”中除第一个“num_batch_dims”维度外的所有元素。
    case:
    x1 = torch.tensor([[1, 2], [3, 4]])
    result1 = SumExceptBatch(x1, num_batch_dims=1)
    >> tensor([3, 7])
    """
    if not is_nonnegative_int(num_batch_dims):
        raise TypeError("Number of batch dimensions must be a non-negative integer.")
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def hwc_and_chw(img):
    """
    将HWC与CHW格式的转换。
    Example:
        hwc_image_numpy = np.random.rand(256, 256, 3)
        chw_image_numpy = hwc_and_chw(hwc_image_numpy)
        print("Original HWC shape:", hwc_image_numpy.shape)
        print("Converted CHW shape:", chw_image_numpy.shape)
        hwc_image_tensor = torch.rand(256, 256, 3)
        chw_image_tensor = hwc_and_chw(hwc_image_tensor)
        print("Original HWC shape:", hwc_image_tensor.shape)
        print("Converted CHW shape:", chw_image_tensor.shape)
    """
    if is_numpy(img):
        if len(img.shape) == 3:
            chwimg = np.rollaxis(img, 2)
            return chwimg
    elif is_tensor(img):
        if len(img.shape) == 3:
            chwimg = img.permute(2, 0, 1).contiguous()
            return chwimg

def to_bchw(tensor):
    """
    转为bchw的格式
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

def image_to_tensor(image, keepdim=True):
    """将numpy图像转换为PyTorch 4d张量图像
        Example:
            img = np.ones((3, 3))
            image_to_tensor(img).shape
            >>> torch.Size([1, 3, 3])

            img = np.ones((4, 4, 1))
            image_to_tensor(img).shape
            >>> torch.Size([1, 4, 4])

            img = np.ones((4, 4, 3))
            image_to_tensor(img, keepdim=False).shape
            >>> torch.Size([1, 3, 4, 4])
    """
    if is_numpy(img):
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
    """将PyTorch张量图像转换为numpy图像
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

def img2tensor(im, totensor=False):
    """NumPy图像数组转换为PyTorch张量"""
    im = np.array(im)
    tensor = np.moveaxis(im, -1, 0).astype(np.float32)
    return torch.from_numpy(tensor).type(torch.FloatTensor) if totensor else tensor

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
    img = np.ones((3, 3))
    print(image_to_tensor(img).shape)
    img = np.ones((4, 4, 1))
    print(image_to_tensor(img).shape)
    img = np.ones((4, 4, 3))
    print(image_to_tensor(img, keepdim=False).shape)
