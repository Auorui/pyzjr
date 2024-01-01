"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is based on PIL implemented transformations.
"""
import torch
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import random
from pyzjr.core import _check_img_is_plt, is_pil,_check_img_is_ndarray, _check_parameter_is_tuple_and_list_or_single_2, \
    _check_input_is_tensor, get_image_size, check_dtype, _get_image_num_channels
from ._utils import Images, Compose
import numpy as np

__all__ = ["PILToTensor", "NdarryToPIL", "TensorToPIL", "MeanStdNormalize", "AdjustBrightness", "AdjustContrast", \
           "AutoContrast", "RandomAutoContrast", "AdjustGamma", "AdjustHue", "AdjustSaturation", "CenterCrop", "EqualizeHistogram", \
           "RandomEqualizeHistogram", "RandomHorizontalFlip", "RandomVerticalFlip", "RandomPCAnoise", "InvertColor", "RandomInvertColor", \
           "Resize", "ColorJitter", "RandomCrop", "RandomRotation", "Grayscale", "RandomGrayscale", "AdjustSharpness", \
           "RandomAdjustSharpness", "GaussianBlur", "ResizedCrop", "RandomResizedCrop", "Pad"]


class PILToTensor():
    """
    Usage Directions:
        Convert a PIL format image to a PyTorch tensor.

    Parameters:
        - half (bool, optional): If True, the resulting tensor will be in half-precision (fp16).
          If False, the tensor will be in single-precision (fp32). Default is False.

    Examples:
        >>> pil_image = Image.open('example.jpg')  # Load a PIL image
        >>> converter = PILToTensor(half=True)
        >>> torch_tensor = converter(pil_image)

    Notes:
        This transformation is used to convert a PIL format image into a PyTorch tensor.
        It supports both single-precision (fp32) and half-precision (fp16) tensors.
    """
    def __init__(self, half=False):
        super(PILToTensor, self).__init__()
        self.half = half

    def pil_to_tensor(self, pil_img):
        img = torch.as_tensor(np.array(pil_img, copy=True))
        img = img.view(pil_img.size[1], pil_img.size[0], len(pil_img.getbands()))
        img = img.permute((2, 0, 1))  # 从HWC格式转换为CHW格式
        img = img.half() if self.half else img.float()  # 将uint8转换为fp16/32
        img /= 255.0
        return img

    def __call__(self, pil_img, *args):
        _check_img_is_plt(pil_img)
        return self.pil_to_tensor(pil_img)


class NdarryToPIL():
    """
    Usage Directions:
        Ensure that the input image is in PIL format, and it is actually a darray to PIL conversion.

    Examples:
        >>> numpy_array = np.random.rand(256, 256, 3)  # Example NumPy array with shape (H, W, C)
        >>> converter = NdarryToPIL()
        >>> pil_image = converter(numpy_array)

    Notes:
        This transformation is used to convert a NumPy array (with proper shape and data type) into a PIL format image.
    """
    def __init__(self):
        super(NdarryToPIL, self).__init__()

    def ndarry_to_pil(self, img):
        if not is_pil(img):
            _check_img_is_ndarray(img)
            if img.ndim not in (2, 3):
                raise ValueError("The dimension of input image should be 2 or 3. Got {}.".format(img.ndim))
            if img.ndim == 2:
                img = np.expand_dims(img, 2)
            if img.shape[-1] > 4:
                raise ValueError("The channel of input image should not exceed 4. Got {}.".format(img.shape[-1]))
            if img.shape[-1] == 1:
                if img.dtype not in (np.bool_, np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32, np.float32,
                                     np.float64):
                    raise TypeError("The input image type {} is not supported when image shape is [H, W] or [H, W, 1].".format(img.dtype))
                img = img[:, :, 0]
            elif img.dtype != np.uint8:
                raise TypeError("The input image type {} is not supported when "
                                "image shape is [H, W, 2], [H, W, 3] or [H, W, 4].".format(img.dtype))
            return Image.fromarray(img)
        return img

    def __call__(self, img):
        return self.ndarry_to_pil(img)

class TensorToPIL():
    """
    Usage Directions:
        Convert a PyTorch tensor to a PIL format image.

    Examples:
        >>> tensor = torch.randn(3, 256, 256)  # Example PyTorch tensor with 3 channels and size 256x256
        >>> converter = TensorToPIL()
        >>> pil_image = converter(tensor)

    Notes:
        This transformation is used to convert a PyTorch tensor (with proper shape and data type) into a PIL format image.
    """
    def __init__(self):
        super(TensorToPIL, self).__init__()

    def tensor_to_pil(self, img):
        if not is_pil(img):
            img = img.clamp(0, 1)  # Ensure values are in the range [0, 1]
            img = img.mul(255).byte()  # Scale to [0, 255] and convert to byte
            img = img.permute(1, 2, 0)  # Rearrange dimensions from CHW to HWC
            return Image.fromarray(img.numpy())
        return img

    def __call__(self, img):
        return self.tensor_to_pil(img)


class MeanStdNormalize():
    """
    Usage Directions:
        This class is used to perform mean-std normalization on the pixel values of a PIL format image.The default is
        to use the mean and variance of Imagenet.

    Parameters:
        - mean (list or tuple): A list or tuple of mean values for each image channel.
        - std (list or tuple): A list or tuple of standard deviation values for each image channel.

    Examples:
        >>> transforms = Compose([PILToTensor(), MeanStdNormalize(), TensorToPIL()])
        >>> transformed_image = transforms(image)

    Notes:
        This transformation normalizes the pixel values of a PIL image based on specified mean and standard deviation values for each channel.
    """
    IMAGENET_MEAN = 0.485, 0.456, 0.406
    IMAGENET_STD = 0.229, 0.224, 0.225
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super(MeanStdNormalize, self).__init__()
        self.mean = mean
        self.std = std

    def mean_std_normalize(self, img):
        if img.ndim < 3:
            raise ValueError(f'Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = {img.size()}.')

        mean = torch.as_tensor(self.mean, dtype=img.dtype, device=img.device)
        std = torch.as_tensor(self.std, dtype=img.dtype, device=img.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(img.dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        img.sub_(mean).div_(std)
        return img

    def __call__(self, img):
        _check_input_is_tensor(img)
        return self.mean_std_normalize(img)

class AdjustBrightness():
    """
    Usage Directions:
        This class is used to adjust the brightness of PIL format images.

    Parameters:
        - brightness_factor (float): A positive number to adjust the brightness.
          Values greater than 1 increase brightness, and values between 0 and 1 decrease brightness.

    Examples:
        >>> transforms = Compose([AdjustBrightness(1.5), PILToTensor()])
        >>> transformed_image = transforms(image)
        >>> transformed_image = torch.clamp(transformed_image, 0, 1)
        >>> transformed_image = transformed_image.permute(1, 2, 0)

    Notes:
        - A high brightness_factor (greater than 1) makes the image brighter.
        - A low brightness_factor (between 0 and 1) makes the image darker.
        - Values outside the range (0, +∞) are accepted.
    """
    def __init__(self, brightness_factor):
        super(AdjustBrightness, self).__init__()
        self.brightness_factor = brightness_factor

    def _brightness(self, pil_img, brightness_factor):
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(brightness_factor)
        return pil_img

    def __call__(self, pil_img, *args):
        _check_img_is_plt(pil_img)
        return self._brightness(pil_img, self.brightness_factor)

class AdjustContrast():
    """
    Usage Directions :
        This class is used to adjust the contrast of a PIL image.

    Parameters:
        - contrast_factor (float): A positive number to adjust the contrast.
          Values greater than 1 increase contrast, and values between 0 and 1 decrease contrast.

    Examples :
        >>> transforms = Compose([AdjustContrast(3), PILToTensor()])
        >>> transformed_image = transforms(image)
        >>> transformed_image = torch.clamp(transformed_image, 0, 1)
        >>> transformed_image = transformed_image.permute(1, 2, 0)

    Notes:
        - High contrast (contrast_factor > 1) makes the image more vivid and enhances the difference between light and dark areas.
        - Low contrast (contrast_factor between 0 and 1) makes the image less vivid and reduces the difference between light and dark areas.
        - Values outside the range (0, +∞) are accepted.
    """
    def __init__(self, contrast_factor):
        super(AdjustContrast, self).__init__()
        self.contrast_factor = contrast_factor

    def _contrast(self, pil_img, contrast_factor):
        enhancer = ImageEnhance.Contrast(pil_img)
        img = enhancer.enhance(contrast_factor)
        return img

    def __call__(self, pil_img):
        _check_img_is_plt(pil_img)
        return self._contrast(pil_img, self.contrast_factor)

class AutoContrast():
    """
    Usage Directions:
        Automatically adjust the contrast of PIL format images.

    Parameters:
        - cutoff (float, optional): A value between 0.0 and 1.0 that controls the fraction of pixels to ignore during contrast adjustment.
          Lower values make the algorithm consider more pixels, which might lead to stronger contrast adjustment.
          Default is 0.0.

        - ignore (int, list, or None, optional): Values to ignore during the contrast adjustment process.
          This can be an integer, a list of integers, or None. Default is None.

    Examples:
        >>> transforms = Compose([AutoContrast(), PILToTensor()])
        >>> transformed_image = transforms(image)
        >>> transformed_image = torch.clamp(transformed_image, 0, 1)
        >>> transformed_image = transformed_image.permute(1, 2, 0)

    Notes:
        This transformation automatically adjusts the contrast of an image based on pixel values.
        You can set the 'cutoff' parameter to control the fraction of extreme pixel values to ignore.
    """
    def __init__(self, cutoff=0.0, ignore=None):
        super(AutoContrast, self).__init__()
        if ignore is None:
            ignore = []
        if isinstance(ignore, int):
            ignore = [ignore]
        self.cutoff = cutoff
        self.ignore = ignore

    def auto_contrast(self,img, cutoff, ignore):
        return ImageOps.autocontrast(img, cutoff, ignore)

    def __call__(self, img):
        _check_img_is_plt(img)
        return self.auto_contrast(img, self.cutoff, self.ignore)

class RandomAutoContrast(AutoContrast):
    """
    Usage Directions:
        Randomly apply the AutoContrast transformation to a PIL format image with a specified probability.

    Example:
        >>> transforms = Compose([RandomAutoContrast(pro=0.5), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)
    """
    def __init__(self, pro=0.5):
        super(RandomAutoContrast, self).__init__()
        self.pro = pro

    def __call__(self, img):
        if random.random() >= self.pro:
            return super().__call__(img,)
        return img

class AdjustGamma():
    """
    Usage Directions:
        Adjust the gamma of a PIL format image.

    Parameters:
        - gamma (float): The gamma value to adjust the image. A higher gamma (>1) makes the image brighter,
          while a lower gamma (<1) makes the image darker.
        - gain (float, optional): A gain factor to adjust the image. Default is 1.

    Examples:
        >>> transforms = Compose([AdjustGamma(3), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)
        >>> plt.imshow(transformed_image)
        >>> plt.axis('off')
        >>> plt.show()

    Notes:
        This transformation adjusts the gamma of a PIL image. A higher gamma value makes the image appear brighter,
        while a lower gamma value makes the image appear darker.
    """
    def __init__(self, gamma, gain=1):
        super(AdjustGamma, self).__init__()
        self.gamma = gamma
        self.gain = gain

    def _gamma(self, img, gamma, gain):
        gamma_table = [(255 + 1 - 1e-3) * gain * pow(x / 255., gamma) for x in range(256)]
        if len(img.split()) == 3:
            gamma_table = gamma_table * 3
            img = img.point(gamma_table)
        elif len(img.split()) == 1:
            img = img.point(gamma_table)
        return img

    def __call__(self, img):
        _check_img_is_plt(img)
        return self._gamma(img, self.gamma, self.gain)

class AdjustHue():
    """
    Usage Directions:
        Adjust the hue of the input image.

    Parameters:
        - hue_factor (float): The hue adjustment factor. Positive values shift the hue towards the red spectrum,
          while negative values shift it towards the blue-green spectrum.

    Example:
        >>> transforms = Compose([AdjustHue(-.5), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)

    Notes:
        This transformation adjusts the hue of a PIL image. The hue adjustment factor controls the direction and
        magnitude of the hue shift.
    """
    def __init__(self, hue_factor):
        super(AdjustHue, self).__init__()
        self.hue_factor = hue_factor

    def _hue(self,img, hue_factor):
        image = img
        image_hue_factor = hue_factor
        if not -0.5 <= image_hue_factor <= 0.5:
            raise ValueError('image_hue_factor {} is not in [-0.5, 0.5].'.format(image_hue_factor))
        mode = image.mode
        if mode in {'L', '1', 'I', 'F'}:
            return image
        hue, saturation, value = img.convert('HSV').split()
        np_hue = np.array(hue, dtype=np.uint8)
        with np.errstate(over='ignore'):
            np_hue += np.uint8(image_hue_factor * 255)
        hue = Image.fromarray(np_hue, 'L')
        image = Image.merge('HSV', (hue, saturation, value)).convert(mode)
        return image

    def __call__(self, img):
        _check_img_is_plt(img)
        return self._hue(img, self.hue_factor)

class AdjustSaturation():
    """
    Usage Directions:
        Adjust the saturation of a PIL format image.

    Parameters:
        - saturation_factor (float): The saturation adjustment factor. Positive values increase the saturation,
          while negative values decrease the saturation.

    Example:
        >>> transforms = Compose([AdjustSaturation(3), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)

    Notes:
        This transformation adjusts the saturation of a PIL image. The saturation adjustment factor controls the
        intensity of colors in the image.
    """
    def __init__(self, saturation_factor):
        super(AdjustSaturation, self).__init__()
        self.saturation_factor = saturation_factor

    def _saturation(self, img, saturation_factor):
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor)
        return img

    def __call__(self, img):
        _check_img_is_plt(img)
        return self._saturation(img, self.saturation_factor)


class CenterCrop():
    """
    Usage Directions:
        Center crop the image.

    Parameters:
        - size (int or tuple): The size of the desired center-cropped image. If an integer is provided, a square crop
          of that size is performed. If a tuple of (width, height) is provided, a rectangular crop with the specified
          dimensions is performed.

    Example:
        >>> transforms = Compose([AdjustSaturation(3), CenterCrop(224) , PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)
        >>> print(transformed_image.size)

    Notes:
        This transformation performs center cropping on a PIL image. The center crop is obtained by selecting a
        rectangular region from the center of the input image.
    """
    def __init__(self, size):
        super(CenterCrop, self).__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def crop(self, img, x, y, width, height):
        return img.crop((x, y, x + width, y + height))

    def center_crop(self,img, size):
        if isinstance(size, int):
            size = (size, size)
        img_width, img_height = img.size
        crop_height, crop_width = size
        crop_y = int(round((img_height - crop_height) / 2.))
        crop_x = int(round((img_width - crop_width) / 2.))
        return self.crop(img, crop_x, crop_y, crop_width, crop_height)

    def __call__(self, img):
        _check_img_is_plt(img)
        return self.center_crop(img, self.size)


class EqualizeHistogram():
    """
    Usage Directions:
        Equalize the histogram of a PIL format image.

    Example:
        >>> transforms = Compose([EqualizeHistogram(), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)

    Notes:
        This transformation equalizes the histogram of a PIL image, which enhances the image's contrast.
    """
    def __init__(self):
        super(EqualizeHistogram, self).__init__()

    def _equalize(self,img):
        return ImageOps.equalize(img)

    def __call__(self, img):
        _check_img_is_plt(img)
        return self._equalize(img)

class RandomEqualizeHistogram(EqualizeHistogram):
    """
    Usage Directions:
        Randomly apply the EqualizeHistogram transformation to the input image with a specified probability.

    Example:
        >>> transforms = Compose([RandomEqualizeHistogram(0.8), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)
    """
    def __init__(self, pro=0.5):
        super(RandomEqualizeHistogram, self).__init__()
        self.pro = pro

    def __call__(self, img):
        if random.random() >= self.pro:
            return super().__call__(img)
        return img

class RandomHorizontalFlip():
    """
    Usage Directions:
        Randomly flip the input pil image horizontally.

    Example:
        >>> transforms = Compose([RandomHorizontalFlip(), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)
    """
    def __init__(self, prob=0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.prob = prob

    def horizontal_flip(self,img):
        return img.transpose(Images.FLIP_LEFT_RIGHT)

    def random_horizontal_flip(self, img, prob):
        if prob > random.random():
            img = self.horizontal_flip(img)
        return img

    def __call__(self, img):
        _check_img_is_plt(img)
        return self.random_horizontal_flip(img, self.prob)

class RandomVerticalFlip():
    """
    Usage Directions:
        Randomly flip the input pil image vertically.

    Example:
        >>> transforms = Compose([RandomVerticalFlip(), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)
    """
    def __init__(self, prob=0.5):
        super(RandomVerticalFlip, self).__init__()
        self.prob = prob

    def vertical_flip(self, img):
        return img.transpose(Images.FLIP_TOP_BOTTOM)

    def random_vertical_flip(self,img, prob):
        if prob > random.random():
            img = self.vertical_flip(img)
        return img

    def __call__(self, img):
        _check_img_is_plt(img)
        return self.random_vertical_flip(img, self.prob)

class RandomPCAnoise():
    """
    Usage Directions:
        Add AlexNet-style PCA-based noise to an image.

    Parameters:
        - alpha (float): The strength of the PCA-based noise. A higher value results in stronger noise.

    Examples:
        >>> transforms = Compose([RandomPCAnoise(3), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)

    Notes:
        This transformation applies PCA-based noise to an image to introduce variations in the image's color and texture.
        The `alpha` parameter controls the strength of the noise.
    """
    def __init__(self, alpha=0.05):
        super(RandomPCAnoise, self).__init__()
        self.alpha = alpha

    def random_pca(self, img, alpha):
        if img.mode != 'RGB':
            img = img.convert("RGB")
        alpha_r = np.random.normal(loc=0.0, scale=alpha)
        alpha_g = np.random.normal(loc=0.0, scale=alpha)
        alpha_b = np.random.normal(loc=0.0, scale=alpha)
        table = np.array([
            [55.46 * -0.5675, 4.794 * 0.7192, 1.148 * 0.4009],
            [55.46 * -0.5808, 4.794 * -0.0045, 1.148 * -0.8140],
            [55.46 * -0.5836, 4.794 * -0.6948, 1.148 * 0.4203]
        ])
        pca_r = table[0][0] * alpha_r + table[0][1] * alpha_g + table[0][2] * alpha_b
        pca_g = table[1][0] * alpha_r + table[1][1] * alpha_g + table[1][2] * alpha_b
        pca_b = table[2][0] * alpha_r + table[2][1] * alpha_g + table[2][2] * alpha_b
        img_arr = np.array(img).astype(np.float64)
        img_arr[:, :, 0] += pca_r
        img_arr[:, :, 1] += pca_g
        img_arr[:, :, 2] += pca_b
        img_arr = np.uint8(np.minimum(np.maximum(img_arr, 0), 255))
        img = Image.fromarray(img_arr)
        return img

    def __call__(self, img):
        _check_img_is_plt(img)
        return self.random_pca(img, self.alpha)

class InvertColor():
    """
    Usage Directions:
        Invert the colors of a PIL format image. i.e: 255-pixel

    Examples:
        >>> transforms = Compose([InvertColor(), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)

    Notes:
        This transformation inverts the colors of a PIL image, which means it changes each color to its complementary color.
    """
    def __init__(self):
        super(InvertColor, self).__init__()

    def _invert_color(self,img):
        return ImageOps.invert(img)

    def __call__(self, img):
        _check_img_is_plt(img)
        return self._invert_color(img)


class RandomInvertColor(InvertColor):
    """
    Usage Directions:
        Randomly invert the colors of a PIL format image with a specified probability.

    Example:
        >>> transforms = Compose([RandomInvertColor(pro=0.5), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)

    """
    def __init__(self, pro=0.5):
        super(RandomInvertColor, self).__init__()
        self.pro = pro

    def __call__(self, img):
        if random.random() >= self.pro:
            return super().__call__(img)
        return img


class Resize():
    """
    Usage Directions:
        Resize the input PIL Image to the given size.

    Parameters:
        - sizes (list and tuple or int): Desired output size. If size is a sequence like (h, w), output size will be
                                        matched to this. If size is an int, smaller edge of the image will be matched
                                        to this number.i.e, if height > width, then image will be rescaled to
                                        (size * height / width, size).

    Examples:
        >>> transforms = Compose([Resize(224), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)
        >>> print(transformed_image.size)
    """
    def __init__(self, sizes):
        super(Resize, self).__init__()
        self.sizes = sizes

    def __WHratio(self, img, sizes):
        self.w, self.h = img.size
        if self.w < self.h:
            ow = sizes
            oh = int(sizes * self.h / self.w)
        else:
            oh = sizes
            ow = int(sizes * self.w / self.h)
        return (ow, oh)

    def resize(self, img, sizes, interpolation=Images.LINEAR):
        _check_parameter_is_tuple_and_list_or_single_2(sizes)
        if isinstance(sizes, int) or len(sizes) == 1:
            if (self.w <= self.h and self.w == sizes) or (self.h <= self.w and self.h == sizes):
                return img
            ratio = self.__WHratio(img, sizes)
            return img.resize(ratio, interpolation)
        else:
            return img.resize(sizes[::-1], interpolation)

    def __call__(self, img):
        _check_img_is_plt(img)
        return self.resize(img, self.sizes)

class ColorJitter():
    """
    Usage Directions:
        Adjusting the brightness, contrast, saturation, and hue of an image.

    Examples:
        >>> transforms = Compose([ColorJitter(1.2,2,1.2,0.3), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)
    """
    def __init__(self, brightness_factor, contrast_factor, saturation_factor, hue_factor):
        super(ColorJitter, self).__init__()
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.saturation_factor = saturation_factor
        self.hue_factor = hue_factor

    def brightness(self, pil_img, brightness_factor):
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(brightness_factor)
        return pil_img

    def contrast(self, pil_img, contrast_factor):
        enhancer = ImageEnhance.Contrast(pil_img)
        img = enhancer.enhance(contrast_factor)
        return img

    def saturation(self, pil_img, saturation_factor):
        enhancer = ImageEnhance.Color(pil_img)
        img = enhancer.enhance(saturation_factor)
        return img

    def hue(self, pil_img, hue_factor):
        image = pil_img
        image_hue_factor = hue_factor
        if not -0.5 <= image_hue_factor <= 0.5:
            raise ValueError('image_hue_factor {} is not in [-0.5, 0.5].'.format(image_hue_factor))
        mode = image.mode
        if mode in {'L', '1', 'I', 'F'}:
            return image
        hue, saturation, value = pil_img.convert('HSV').split()
        np_hue = np.array(hue, dtype=np.uint8)
        with np.errstate(over='ignore'):
            np_hue += np.uint8(image_hue_factor * 255)
        hue = Image.fromarray(np_hue, 'L')
        image = Image.merge('HSV', (hue, saturation, value)).convert(mode)
        return image

    def _jitter(self, pil_img, brightness_factor, contrast_factor, saturation_factor, hue_factor):
        brightness_img = self.brightness(pil_img, brightness_factor)
        contrast_img = self.contrast(brightness_img, contrast_factor)
        saturation_img = self.saturation(contrast_img, saturation_factor)
        hue_img = self.hue(saturation_img, hue_factor)
        return hue_img

    def __call__(self, pil_img):
        _check_img_is_plt(pil_img)
        return self._jitter(pil_img, self.brightness_factor, self.contrast_factor, self.saturation_factor, self.hue_factor)

class RandomCrop():
    """
    Usage Directions:
        Random cropping on a PIL format image.

    Parameters:
        - size (int or tuple): The output size of the cropped image. If an integer is provided, the output will be a square crop.

    Examples:
        >>> transforms = Compose([RandomCrop(224), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)
        >>> print(transformed_image.size)
    """
    def __init__(self, size):
        super(RandomCrop, self).__init__()
        self.size = size

    def get_shape(self, image, output_size):
        if isinstance(output_size, int):
            output_size = [output_size, output_size]
        h, w = get_image_size(image)
        outw, outh = output_size
        if w == outw and h == outh:
            return 0, 0, h, w

        if h + 1 < outh or w + 1 < outw:
            raise ValueError(f"Required crop size {(outh, outw)} is larger then input image size {(h, w)}")
        i = torch.randint(0, h - outh + 1, size=(1, )).item()
        j = torch.randint(0, w - outw + 1, size=(1, )).item()
        return i, j, outh, outw

    def crop(self, img, x, y, width, height):
        return img.crop((x, y, x + width, y + height))

    def __call__(self, image):
        _check_img_is_plt(image)
        i, j, h, w = self.get_shape(image, self.size)
        return self.crop(image, j, i, w, h)  # 注意这里坐标的顺序


class RandomRotation():
    """
    Usage Directions:
        Rotate the input PIL Image by a random angle.

    Examples:
        >>> transforms = Compose([RandomRotation(), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)

    Notes:
        This transformation rotates the input PIL image by a random angle selected from [0°, 90°, 180°, 270°].
    """
    def __init__(self):
        super(RandomRotation, self).__init__()

    def rotate(self, img, angle):
        rotated_img = img.rotate(angle)
        return rotated_img

    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return self.rotate(img, angle)

class Grayscale():
    """
    Convert a PIL format image to grayscale.

    Example:
        >>> transforms = Compose([Grayscale(), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)
    """
    def __init__(self):
        super(Grayscale, self).__init__()

    def to_gray(self, img):
        channels = _get_image_num_channels(img)
        if channels == 1:
            img = img.convert('L')
        elif channels == 3:
            img = img.convert('L')
            np_img = np.array(img, dtype=np.uint8)
            np_img = np.dstack([np_img, np_img, np_img])
            img = Image.fromarray(np_img, 'RGB')
        return img

    def __call__(self, img):
        _check_img_is_plt(img)
        return self.to_gray(img)

class RandomGrayscale(Grayscale):
    """
    Usage Directions:
        Randomly apply the Grayscale transformation to a PIL format image with a specified probability.

    Example:
        >>> transforms = Compose([RandomGrayscale(pro=0.5), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)
    """
    def __init__(self, pro=0.5):
        super(RandomGrayscale, self).__init__()
        self.pro = pro

    def __call__(self, img):
        if random.random() >= self.pro:
            return super().__call__(img)
        return img

class AdjustSharpness():
    """
    Usage Directions:
        Adjust the sharpness of a PIL format image.

    Parameters:
        sharpness_factor (float): The factor to adjust sharpness. A value less than 1.0 decreases sharpness, while a
        value greater than 1.0 increases sharpness. 1.0 represents the original sharpness.

    Example:
        >>> transforms = Compose([AdjustSharpness(2.0), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)
    """
    def __init__(self, sharpness_factor):
        super(AdjustSharpness, self).__init__()
        self.sharpness_factor = sharpness_factor

    def adjust_sharpness(self, img, sharpness_factor):
        _check_img_is_plt(img)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness_factor)
        return img

    def __call__(self, img):
        return self.adjust_sharpness(img,self.sharpness_factor)

class RandomAdjustSharpness(AdjustSharpness):
    """
    Usage Directions:
        Randomly adjust the clarity of the input image
    Example:
        >>> transforms = Compose([RandomAdjustSharpness(4.0), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)
    """
    def __init__(self, sharpness_factor, pro=0.5):
        super(RandomAdjustSharpness, self).__init__(sharpness_factor)  # 修正super()参数，调用父类的__init__
        self.sharpness_factor = sharpness_factor
        self.pro = pro

    def __call__(self, img):
        if random.random() >= self.pro:
            return super().__call__(img)
        return img

class GaussianBlur():
    """
    Usage Directions:
        Apply Gaussian blur to a PIL format image.

    Parameters:
        - radius (int): The radius of the Gaussian blur kernel. A larger value results in a stronger blur effect.

    Example:
        >>> transforms = Compose([GaussianBlur(2), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)
    """
    def __init__(self, radius):
        self.radius = radius

    def apply_gaussian_blur(self, img, radius):
        type = check_dtype(radius)
        if type.is_int():
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius))
            return blurred_img

    def __call__(self, img):
        _check_img_is_plt(img)
        return self.apply_gaussian_blur(img, self.radius)


class Pad():
    """
    Usage Directions:
        Apply padding to a PIL format image.

    Parameters:
        - padding (int, tuple, or list): The padding to be applied. It can be an integer for uniform padding or a tuple/list
          of size 2 or 4 for specifying padding on each side separately.
        - padding_mode (str): The padding mode, which should be one of 'constant', 'edge', 'reflect', or 'symmetric'.
        - fill_value (tuple): The fill value used when padding_mode is 'constant'. This should be a tuple specifying the fill
          color in the format (R, G, B).

    Examples:
        >>> transforms = Compose([Pad(10), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)
    """
    def __init__(self, padding, padding_mode='constant',fill_value=(128, 128, 128)):
        super(Pad, self).__init__()
        self.padding = padding
        self.padding_mode = padding_mode
        self.fill_value = fill_value
        if self.padding_mode not in ['constant', 'edge', 'reflect', 'symmetric']:
            raise ValueError("Padding mode should be 'constant', 'edge', 'reflect', or 'symmetric'.")

    def pad(self, img, padding, padding_mode, fill_value):
        type = check_dtype(padding)
        if type.is_int():
            top = bottom = left = right = padding

        elif type.is_list_or_tuple():
            if len(padding) == 2:
                left = right = padding[0]
                top = bottom = padding[1]
            elif len(padding) == 4:
                left, top, right, bottom = padding
            else:
                raise ValueError("The size of the padding list or tuple should be 2 or 4.")
        else:
            raise TypeError("Padding can be any of: a number, a tuple or list of size 2 or 4.")
        if padding_mode == 'constant':
            if img.mode == 'P':
                palette = img.getpalette()
                image = ImageOps.expand(img, border=(left, top, right, bottom), fill=fill_value)
                image.putpalette(palette)
                return image
            if isinstance(fill_value, tuple) and (img.mode == 'L' or img.mode == '1'):
                fill_value = (fill_value[0],)
            return ImageOps.expand(img, border=(left, top, right, bottom), fill=fill_value)
        if img.mode == 'P':
            palette = img.getpalette()
            img = np.asarray(img)
            img = np.pad(img, ((top, bottom), (left, right)), padding_mode)
            img = Image.fromarray(img)
            img.putpalette(palette)
            return img
        img = np.asarray(img)
        if len(img.shape) == 3:
            img = np.pad(img, ((top, bottom), (left, right), (0, 0)), padding_mode)
        if len(img.shape) == 2:
            img = np.pad(img, ((top, bottom), (left, right)), padding_mode)
        return Image.fromarray(img)

    def __call__(self, pil_img):
        _check_img_is_plt(pil_img)
        return self.pad(pil_img,self.padding,self.padding_mode,self.fill_value)

class ResizedCrop(Resize):
    """
    Usage Directions:
        Crop and resize a PIL format image to the specified size.

    Parameters:
        - sizes (tuple): The target size for resizing in the format (width, height).
        - box (tuple): A tuple in the format (x, y, width, height) specifying the crop box coordinates.

    Examples:
        >>> transforms = Compose([ResizedCrop(224, (10, 10, 100, 100)), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)
    """
    def __init__(self, sizes, box):
        super(ResizedCrop, self).__init__(sizes)
        type = check_dtype(sizes)
        if type.is_int():
            self.sizes = [sizes, sizes]
        self.box = box

    def crop_img(self, img, x, y, width, height):
        return img.crop((x, y, x + width, y + height))

    def resize_img(self, img, size):
        return super().resize(img, size)

    def resize_crop(self, img, x, y, width, height, size):
        img = self.crop_img(img, x, y, width, height)
        img = self.resize_img(img, size)
        return img

    def __call__(self, img):
        _check_img_is_plt(img)
        if len(self.box) == 4:
            x, y, width, height = self.box
            return self.resize_crop(img, x, y, width, height, self.sizes)
        else:
            raise ValueError("Box should be a tuple of 4 integers (x, y, width, height).")


class RandomResizedCrop(ResizedCrop):
    """
    Usage Directions:
        Randomly crop and resize a PIL format image.

    Parameters:
        - sizes (tuple): The target size for resizing in the format (width, height).
        - scale (tuple): A tuple (min_scale, max_scale) specifying the range for random scaling.
        - box (tuple, optional): A tuple (x, y, width, height) specifying a fixed crop box. If None, a random box will be generated.

    Examples:
        >>> transforms = Compose([RandomResizedCrop(224, (0.2, 1.0)), PILToTensor(), TensorToPIL()])
        >>> transformed_image = transforms(image)
    """
    def __init__(self, sizes, scale=(0.2, 1.0), box=None):
        super(RandomResizedCrop, self).__init__(sizes,box)
        self.scale = scale
        self.box = box

    def get_random_box(self, image, scale):
        width, height = image.size
        minscale_img = (width * scale[0], height * scale[0])
        maxscale_img = (width * scale[1], height * scale[1])
        w = random.randint(minscale_img[0], maxscale_img[0])
        h = random.randint(minscale_img[1], maxscale_img[1])
        if w <= width and h <= height:
            x = random.randint(0, width - w)
            y = random.randint(0, height - h)
            return x, y, w, h
        else:
            return 0, 0, width, height

    def __call__(self, img):
        _check_img_is_plt(img)
        if self.box is None:
            x, y, w, h = self.get_random_box(img, self.scale)
        else:
            x, y, w, h = self.box
        return super().resize_crop(img, x, y, w, h, self.sizes)

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("TkAgg")
    image = Image.open(r'D:\PythonProject\Torchproject\classification\dataset\crack\1181.png')

    transforms = Compose([RandomResizedCrop(224), PILToTensor(), TensorToPIL()])
    transformed_image = transforms(image)

    print(transformed_image.size)
    # transformed_image = torch.clamp(transformed_image, 0, 1)
    # transformed_image = transformed_image.permute(1, 2, 0)

    plt.imshow(transformed_image)
    plt.axis('off')
    plt.show()
