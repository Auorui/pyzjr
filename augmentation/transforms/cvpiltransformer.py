"""
Copyright (c) 2024, Auorui.
All rights reserved.

Image enhancement suitable for OpenCV and Pillow
"""
import cv2
import torch
import numpy as np
import random
from math import ceil
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from pyzjr.utils.check import is_pil, is_numpy, get_image_size, get_image_num_channels, \
    is_positive_int, is_list_or_tuple
from pyzjr.augmentation.transforms.applytransformer import Images
from pyzjr.utils.randfun import rand

class ToTensor(object):
    """转换为Tensor格式"""
    def __init__(self, half=False):
        super(ToTensor, self).__init__()
        self.half = half

    def pil_to_tensor(self, pil_img):
        img = torch.as_tensor(np.array(pil_img, copy=True))
        img = img.view(pil_img.size[1], pil_img.size[0], len(pil_img.getbands()))
        img = img.permute((2, 0, 1))  # 从HWC格式转换为CHW格式
        return img

    def cv_to_tensor(self, np_img):
        np_img = np_img[:, :, ::-1]
        img = np.ascontiguousarray(np_img.transpose((2, 0, 1)))  # BGR to RGB -> HWC to CHW -> contiguous
        img = torch.from_numpy(img)
        return img

    def __call__(self, image):
        if is_pil(image):
            image = self.pil_to_tensor(image)
        elif is_numpy(image):
            image = self.cv_to_tensor(image)
        img = image.half() if self.half else image.float()  # 将uint8转换为fp16/32
        return img

class RandomHorizontalFlip(object):
    """该类用于随机水平翻转"""
    def __init__(self, pro=0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.p = pro
        self.random_float = random.random()

    def pil_hflip(self, img):
        return img.transpose(Images.FLIP_LEFT_RIGHT)

    def cv_hflip(self, img):
        return cv2.flip(img, 1)

    def __call__(self, image):
        if self.random_float >= self.p:
            if is_pil(image):
                image = self.pil_hflip(image)
            elif is_numpy(image):
                image = self.cv_hflip(image)
        return image

class RandomVerticalFlip():
    """该类用于随机垂直翻转"""
    def __init__(self, pro=0.5):
        super(RandomVerticalFlip, self).__init__()
        self.p = pro
        self.random_float = random.random()

    def pil_vflip(self, img):
        return img.transpose(Images.FLIP_TOP_BOTTOM)

    def cv_vflip(self, img):
        return cv2.flip(img, 0)

    def __call__(self, image):
        if self.random_float >= self.p:
            if is_pil(image):
                image = self.pil_vflip(image)
            elif is_numpy(image):
                image = self.cv_vflip(image)
        return image

class AdjustBrightness(object):
    """该类用于调制亮度"""
    def __init__(self, brightness_factor):
        super(AdjustBrightness, self).__init__()
        self.brightness_factor = brightness_factor

    def pil_brightness(self, pil_img, brightness_factor):
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(brightness_factor)
        return pil_img

    def cv_brightness(self, np_img, brightness_factor):
        opencv_img = cv2.convertScaleAbs(np_img, alpha=brightness_factor, beta=0)
        return opencv_img

    def __call__(self, image):
        if is_pil(image):
            image = self.pil_brightness(image, self.brightness_factor)
        elif is_numpy(image):
            image = self.cv_brightness(image, self.brightness_factor)
        return image

class RandomAdjustBrightness(AdjustBrightness):
    """默认50%的概率使用调制图像亮度"""
    def __init__(self, factor, pro=0.5):
        super(RandomAdjustBrightness, self).__init__(factor)  # 修正super()参数，调用父类的__init__
        self.pro = pro

    def __call__(self, img):
        if random.random() >= self.pro:
            return super().__call__(img)
        return img

class AdjustGamma(object):
    """该类用于调整图像的伽玛值"""
    def __init__(self, gamma, gain=1):
        super(AdjustGamma, self).__init__()
        self.gamma = gamma
        self.gain = gain

    def pil_gamma(self, img, gamma, gain):
        gamma_table = [(255 + 1 - 1e-3) * gain * pow(x / 255., gamma) for x in range(256)]
        if len(img.split()) == 3:
            gamma_table = gamma_table * 3
            img = img.point(gamma_table)
        elif len(img.split()) == 1:
            img = img.point(gamma_table)
        return img

    def cv_gamma(self, img, gamma, gain):
        img_np = np.array(img)
        img_gamma_corrected = ((img_np / 255.0) ** gamma) * 255.0 * gain
        img_gamma_corrected = np.clip(img_gamma_corrected, 0, 255).astype(np.uint8)
        return img_gamma_corrected

    def __call__(self, image):
        if is_pil(image):
            image = self.pil_gamma(image, self.gamma, self.gain)
        elif is_numpy(image):
            image = self.cv_gamma(image, self.gamma, self.gain)
        return image

class RandomAdjustGamma(AdjustGamma):
    """默认50%的概率使用调整图像的伽玛值"""
    def __init__(self, gamma, gain=1, pro=0.5):
        super(RandomAdjustGamma, self).__init__(gamma=gamma, gain=gain)  # 修正super()参数，调用父类的__init__
        self.pro = pro

    def __call__(self, img):
        if random.random() >= self.pro:
            return super().__call__(img)
        return img

class ToHSV(object):
    """该类用于对图像应用HSV增强"""
    def __init__(self, hue_range=(-0.1, 0.1), saturation_range=(0.6, 1.4), value_range=(0.7, 1.3)):
        super(ToHSV, self).__init__()
        self.hue_range = hue_range
        self.saturation_range = saturation_range
        self.value_range = value_range

    def _hsv(self, hsv_image):
        hue_factor = rand(self.hue_range[0], self.hue_range[1])
        saturation_factor = rand(self.saturation_range[0], self.saturation_range[1])
        value_factor = rand(self.value_range[0], self.value_range[1])

        hsv_image[:, :, 0] = np.clip(hsv_image[:, :, 0] * hue_factor, 0, 179)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * value_factor, 0, 255)
        return hsv_image

    def __call__(self, image):
        if is_pil(image):
            hsv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
            hsv_image = self._hsv(hsv_image)
            _image = Image.fromarray(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB))
        if is_numpy(image):
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_image = self._hsv(hsv_image)
            _image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return _image

class RandomRotation():
    """该类用于对图像进行随机旋转"""
    def __init__(self, degrees=None):
        super(RandomRotation, self).__init__()
        self.degrees = degrees

    def pil_rotate(self, img, angle):
        return img.rotate(angle)

    def cv_rotate(self, image, angle, fill_value=(0, 0, 0)):
        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=fill_value)
        return rotated_image

    def __call__(self, image):
        h, w = get_image_size(image)
        is_squre = True if h == w else False
        if self.degrees is not None:
            angle = random.uniform(self.degrees[0], self.degrees[1])
        else:
            if is_squre:
                angle = random.choice([0, 90, 180, 270])
            else:
                angle = random.choice([0, 180])
        if is_pil(image):
            image = self.pil_rotate(image, angle)
        elif is_numpy(image):
            image = self.cv_rotate(image, angle)
        return image

class GaussianBlur():
    """该类用于对图像进行高斯模糊"""
    def __init__(self, ksize:int):
        self.ksize = ksize
        if self.ksize not in {1, 3, 5, 7, 9}:
            raise ValueError(f"Invalid ksize value {ksize}.The available values are 1, 3, 5, 7, 9")

    def cv_blur(self, np_img):
        sigma = 0.3 * ((self.ksize - 1) * 0.5 - 1) + 0.8
        np_img = cv2.GaussianBlur(np_img, ksize=(self.ksize, self.ksize), sigmaX=sigma)
        return np_img

    def pil_blur(self, pil_img):
        pillow_img = pil_img.filter(ImageFilter.GaussianBlur(radius=self.ksize))
        return pillow_img

    def __call__(self, image):
        if is_pil(image):
            image = self.pil_blur(image)
        elif is_numpy(image):
            image = self.cv_blur(image)
        return image

class Grayscale():
    """该类用于灰度化"""
    def __init__(self):
        super(Grayscale, self).__init__()

    def pil_to_gray(self, img):
        channels = get_image_num_channels(img)
        if channels == 3:
            img = img.convert('L')
            np_img = np.array(img, dtype=np.uint8)
            np_img = np.dstack([np_img, np_img, np_img])
            img = Image.fromarray(np_img, 'RGB')
        return img

    def cv_to_gray(self, img):
        channels = get_image_num_channels(img)
        if channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def __call__(self, image):
        if is_pil(image):
            image = self.pil_to_gray(image)
        elif is_numpy(image):
            image = self.cv_to_gray(image)
        return image

class EqualizeHistogram():
    """该类用于对图像进行直方图均衡化"""
    def __init__(self, clahe=True):
        self.clahe = clahe

    def cv_equalize(self, img):
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        if self.clahe:
            c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            yuv[:, :, 0] = c.apply(yuv[:, :, 0])
        else:
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def pil_equalize(self, img):
        return ImageOps.equalize(img)

    def __call__(self, image):
        if is_pil(image):
            image = self.pil_equalize(image)
        elif is_numpy(image):
            image = self.cv_equalize(image)
        return image

class RandomEqualizeHistogram(EqualizeHistogram):
    """默认50%的概率使用直方图均衡化"""
    def __init__(self, pro=0.5, clahe=True):
        super(RandomEqualizeHistogram, self).__init__(clahe=clahe)
        self.pro = pro

    def __call__(self, img):
        if random.random() >= self.pro:
            return super().__call__(img)
        return img

class RandomLighting():
    """该类用于对图像进行随机光照调整"""
    def __init__(self, alpha):
        self.alpha = alpha

    def cv_lighting(self, image):
        alpha_b = np.random.normal(loc=0.0, scale=self.alpha)
        alpha_g = np.random.normal(loc=0.0, scale=self.alpha)
        alpha_r = np.random.normal(loc=0.0, scale=self.alpha)
        table = np.array([
            [55.46 * -0.5675, 4.794 * 0.7192, 1.148 * 0.4009],
            [55.46 * -0.5808, 4.794 * -0.0045, 1.148 * -0.8140],
            [55.46 * -0.5836, 4.794 * -0.6948, 1.148 * 0.4203]
        ])
        pca_b = table[2][0] * alpha_r + table[2][1] * alpha_g + table[2][2] * alpha_b
        pca_g = table[1][0] * alpha_r + table[1][1] * alpha_g + table[1][2] * alpha_b
        pca_r = table[0][0] * alpha_r + table[0][1] * alpha_g + table[0][2] * alpha_b
        img_arr = np.array(image).astype(np.float64)
        img_arr[:, :, 0] += pca_b
        img_arr[:, :, 1] += pca_g
        img_arr[:, :, 2] += pca_r
        img_arr = np.uint8(np.minimum(np.maximum(img_arr, 0), 255))

        return img_arr

    def pil_lighting(self, image):
        factor = 1.0 + random.uniform(-self.alpha, self.alpha)
        factor = max(0.1, min(factor, 10.0))
        enhancer = ImageEnhance.Brightness(image)
        adjusted_image = enhancer.enhance(factor)

        return adjusted_image

    def __call__(self, image):
        if is_pil(image):
            image = self.pil_lighting(image)
        elif is_numpy(image):
            image = self.cv_lighting(image)
        return image

class Centerzoom():
    """该类用于中心缩放"""
    def __init__(self, factor):
        super().__init__()
        assert factor >= 1, "Zoom factor must be greater than or equal to 1"
        self.factor = factor

    def cv_centerzoom(self, image):
        h, w = image.shape[:2]
        h_ch, w_ch = ceil(h / self.factor), ceil(w / self.factor)
        h_top, w_top = (h - h_ch) // 2, (w - w_ch) // 2
        zoomed_img = cv2.resize(image[h_top: h_top + h_ch, w_top: w_top + w_ch], (w, h), interpolation=cv2.INTER_LINEAR)

        return zoomed_img

    def pil_centerzoom(self, image):
        width, height = image.size
        new_width, new_height = ceil(width / self.factor), ceil(height / self.factor)
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = (width + new_width) // 2
        bottom = (height + new_height) // 2

        cropped_img = image.crop((left, top, right, bottom))
        zoomed_img = cropped_img.resize((width, height), Image.LANCZOS)

        return zoomed_img

    def __call__(self, image):
        if is_pil(image):
            image = self.pil_centerzoom(image)
        elif is_numpy(image):
            image = self.cv_centerzoom(image)
        return image

class Resize(object):
    """该类用于调整图像大小到指定的尺寸"""
    def __init__(self, target_size):
        super(Resize, self).__init__()
        if is_positive_int(target_size):
            self.h = self.w = target_size
        elif is_list_or_tuple(target_size) and len(target_size) == 2:
            self.w, self.h = target_size
        self.target_size = target_size

    def pil_resize(self, img, interpolation=Images.LINEAR):
        img_resized = img.resize((self.w, self.h), interpolation)
        return img_resized

    def cv_resize(self, img, interpolation=cv2.INTER_CUBIC):
        image = cv2.resize(img, (self.w, self.h), interpolation=interpolation)
        return image

    def __call__(self, img):
        if is_pil(img):
            img = self.pil_resize(img)
        elif is_numpy(img):
            img = self.cv_resize(img)
        return img

class RandomCrop(object):
    """该类用于随机裁剪出一个指定大小的区域。"""
    def __init__(self, target_size):
        super(RandomCrop, self).__init__()
        if is_positive_int(target_size):
            self.h = self.w = target_size
        elif is_list_or_tuple(target_size) and len(target_size) == 2:
            self.w, self.h = target_size
        self.target_size = target_size

    def pil_random_crop(self, img):
        width, height = img.size
        if width < self.w or height < self.h:
            raise ValueError("Image size should be larger than target size.")

        left = random.randint(0, width - self.w)
        top = random.randint(0, height - self.h)
        right = left + self.w
        bottom = top + self.h

        cropped_img = img.crop((left, top, right, bottom))
        return cropped_img

    def cv_random_crop(self, img):
        h, w = img.shape[:2]
        if w < self.w or h < self.h:
            raise ValueError("Image size should be larger than target size.")
        x = random.randint(0, w - self.w)
        y = random.randint(0, h - self.h)
        cropped_img = img[y:y+self.h, x:x+self.w]
        return cropped_img

    def __call__(self, image):
        if is_pil(image):
            image = self.pil_random_crop(image)
        elif is_numpy(image):
            image = self.cv_random_crop(image)
        return image

class RandomResizeCrop(RandomCrop):
    """类用于随机调整图像大小并裁剪出一个指定大小的区域。"""
    def __init__(self, target_size, scale_range=(1., 2.)):
        super(RandomResizeCrop, self).__init__(target_size=target_size)
        if scale_range[0] < 1.0 or scale_range[1] < 1.0:
            raise ValueError("Scale range must have values greater than or equal to 1.0")
        if scale_range[0] > scale_range[1]:
            raise ValueError("The lower bound of scale_range must be smaller than the upper bound")
        self.scale_range = scale_range
        self.scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])

    def pil_random_resize(self, img):
        width, height = img.size
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        return resized_img

    def cv_random_resize(self, img):
        h, w = img.shape[:2]
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return resized_img

    def __call__(self, image):
        if is_pil(image):
            image = self.pil_random_resize(image)
            image = self.pil_random_crop(image)
        if is_numpy(image):
            image = self.cv_random_resize(image)
            image = self.cv_random_crop(image)
        return image

class CenterCrop():
    """该类用于从图像中心裁剪出一个指定大小的区域。"""
    def __init__(self, target_size):
        super(CenterCrop, self).__init__()
        if is_positive_int(target_size):
            self.h = self.w = target_size
        elif is_list_or_tuple(target_size) and len(target_size) == 2:
            self.w, self.h = target_size
        self.target_size = target_size

    def pil_center_crop(self, img):
        img_width, img_height = img.size

        left = (img_width - self.w) // 2
        top = (img_height - self.h) // 2
        right = left + self.w
        bottom = top + self.h

        return img.crop((left, top, right, bottom))

    def cv_center_crop(self, img):
        img_height, img_width = img.shape[:2]

        start_x = (img_width - self.w) // 2
        start_y = (img_height - self.h) // 2
        end_x = start_x + self.w
        end_y = start_y + self.h

        return img[start_y:end_y, start_x:end_x]

    def __call__(self, image):
        if is_pil(image):
            image = self.pil_center_crop(image)
        elif is_numpy(image):
            image = self.cv_center_crop(image)
        return image

class Pad():
    """该类用于对图像进行边缘填充。"""
    def __init__(self, padding, fill_value=(128, 128, 128)):
        super(Pad, self).__init__()
        if is_positive_int(padding):
            top = bottom = left = right = padding

        elif is_list_or_tuple(padding):
            if len(padding) == 2:
                left = right = padding[0]
                top = bottom = padding[1]
            elif len(padding) == 4:
                left, top, right, bottom = padding
            else:
                raise ValueError("The size of the padding list or tuple should be 2 or 4.")
        else:
            raise TypeError("Padding can be any of: a number, a tuple or list of size 2 or 4.")
        self.paddings = left, top, right, bottom
        self.fill_value = fill_value

    def pil_pad(self, img):
        left, top, right, bottom = self.paddings
        img_padded = ImageOps.expand(img, (left, top, right, bottom), self.fill_value)
        return img_padded

    def cv_pad(self, img):
        left, top, right, bottom = self.paddings
        img_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.fill_value)
        return img_padded

    def __call__(self, image):
        if is_pil(image):
            image = self.pil_pad(image)
        elif is_numpy(image):
            image = self.cv_pad(image)
        return image

class ResizePad():
    """
    该类用于调整图像大小并进行填充。
    提供了两种图像处理模式：调整大小后填充（frame=True）或直接调整大小（frame=False）。
    """
    def __init__(self, target_size, frame=True):
        super(ResizePad, self).__init__()
        if is_positive_int(target_size):
            self.h = self.w = target_size
        elif is_list_or_tuple(target_size) and len(target_size) == 2:
            self.w, self.h = target_size
        self.frame = frame

    def pil_resizepad(self, image):
        w, h = self.w, self.h
        iw, ih = image.size
        if self.frame:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, ((w-nw) // 2, (h-nh) // 2))
            return new_image
        else:
            return image.resize((w, h), Image.BICUBIC)

    def cv_resizepad(self, image):
        ih, iw = image.shape[:2]
        w, h = self.w, self.h
        if self.frame:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
            new_image = np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)
            top = (h - nh) // 2
            left = (w - nw) // 2
            new_image[top:top + nh, left:left + nw] = resized_image
            return new_image
        else:
            return cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)

    def __call__(self, image):
        if is_pil(image):
            image = self.pil_resizepad(image)
        if is_numpy(image):
            image = self.cv_resizepad(image)
        return image

class InvertColor():
    """该类用于对图像进行颜色反转，即将图像的像素颜色值从原色反转为补色。"""
    def __init__(self):
        super(InvertColor, self).__init__()

    def pil_invert_color(self, img):
        return ImageOps.invert(img)

    def cv_invert_color(self, img):
        return 255 - img

    def __call__(self, image):
        if is_pil(image):
            image = self.pil_invert_color(image)
        elif is_numpy(image):
            image = self.cv_invert_color(image)
        return image

class RandomInvertColor(InvertColor):
    """默认50%概率使用对图像进行颜色反转"""
    def __init__(self, pro=0.5):
        super(RandomInvertColor, self).__init__()
        self.pro = pro

    def __call__(self, img):
        if random.random() >= self.pro:
            return super().__call__(img)
        return img

class AdjustSharpness():
    """
    该类用于调整图像的锐度。锐度调整可以增强图像的边缘细节，使图像看起来更加清晰或更具纹理。
    锐度因子，> 1表示增加锐度，< 1表示降低锐度。
    """
    def __init__(self, factor):
        super(AdjustSharpness, self).__init__()
        self.sharpness_factor = factor

    def pil_adjust_sharpness(self, img):
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(self.sharpness_factor)
        return img

    def cv_adjust_sharpness(self, img):
        kernel = np.array([[0, -1, 0],
                           [-1,  5, -1],
                           [0, -1, 0]])
        # kernel = kernel * self.sharpness_factor
        sharpened_img = cv2.filter2D(img, -1, kernel)
        return sharpened_img

    def __call__(self, image):
        if is_pil(image):
            image = self.pil_adjust_sharpness(image)
        elif is_numpy(image):
            image = self.cv_adjust_sharpness(image)
        return image

class RandomAdjustSharpness(AdjustSharpness):
    """默认50%概率使用调整图像的锐度"""
    def __init__(self, factor, pro=0.5):
        super(RandomAdjustSharpness, self).__init__(factor)  # 修正super()参数，调用父类的__init__
        self.sharpness_factor = factor
        self.pro = pro

    def __call__(self, img):
        if random.random() >= self.pro:
            return super().__call__(img)
        return img

class AdjustSaturation():
    """
    该类用于调整图像的饱和度，可以通过指定一个饱和度因子来增强或降低图像的饱和度。
    饱和度影响图像中色彩的浓淡，值越高，图像的颜色越鲜艳；值越低，图像的颜色越灰。
    """
    def __init__(self, factor):
        super(AdjustSaturation, self).__init__()
        self.saturation_factor = factor

    def pil_saturation(self, img):
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(self.saturation_factor)
        return img

    def cv_saturation(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_img = np.array(hsv_img, dtype=np.float32)
        hsv_img[:, :, 1] = hsv_img[:, :, 1] * self.saturation_factor
        hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
        img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img

    def __call__(self, image):
        if is_pil(image):
            image = self.pil_saturation(image)
        elif is_numpy(image):
            image = self.cv_saturation(image)
        return image

class RandomAdjustSaturation(AdjustSaturation):
    """默认50%概率使用调整图像的饱和度"""
    def __init__(self, factor, pro=0.5):
        super(RandomAdjustSaturation, self).__init__(factor)  # 修正super()参数，调用父类的__init__
        self.pro = pro

    def __call__(self, img):
        if random.random() >= self.pro:
            return super().__call__(img)
        return img

class ColorJitter():
    """
    该类用于对图像进行随机颜色变化处理，包括亮度、对比度、饱和度和色相的随机扰动。
    这些操作常用于数据增强，以增加训练数据的多样性，从而提高模型的泛化能力。
    """
    def __init__(self,
                 brightness_factor_range=(0.5, 1.5),  # 亮度范围
                 contrast_factor_range=(0.5, 1.5),    # 对比度范围
                 saturation_factor_range=(0.5, 1.5),  # 饱和度范围
                 hue_factor_range=(-0.1, 0.1)         # 色相范围
                 ):
        super(ColorJitter, self).__init__()
        self.brightness_factor_range = brightness_factor_range
        self.contrast_factor_range = contrast_factor_range
        self.saturation_factor_range = saturation_factor_range
        self.hue_factor_range = hue_factor_range

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

        # Correct hue wrapping
        np_hue = (np_hue + np.uint8(image_hue_factor * 255)) % 256

        hue = Image.fromarray(np_hue, 'L')
        image = Image.merge('HSV', (hue, saturation, value)).convert(mode)
        return image

    def _jitter(self, pil_img):
        brightness_factor = random.uniform(*self.brightness_factor_range)
        contrast_factor = random.uniform(*self.contrast_factor_range)
        saturation_factor = random.uniform(*self.saturation_factor_range)
        hue_factor = random.uniform(*self.hue_factor_range)
        # Apply jitter in the correct order
        img = self.brightness(pil_img, brightness_factor)
        img = self.contrast(img, contrast_factor)
        img = self.saturation(img, saturation_factor)
        img = self.hue(img, hue_factor)
        return img

    def __call__(self, image):
        if is_numpy(image):
            image = Image.fromarray(image[:, :, ::-1])
            image = self._jitter(image)
            image = np.array(image)[:, :, ::-1]
        elif is_pil(image):
            image = self._jitter(image)
        return image

class MeanStdNormalize():
    """ 应放在 ToTensor 之后
    该类用于对图像进行基于均值（mean）和标准差（std）的归一化操作。
    如果没有提供均值和标准差，它将默认使用ImageNet数据集的常见均值和标准差进行归一化。
    """
    def __init__(self, mean=None, std=None):
        super(MeanStdNormalize, self).__init__()
        self.IMAGENET_MEAN = 0.485, 0.456, 0.406
        self.IMAGENET_STD = 0.229, 0.224, 0.225
        self.mean = self.IMAGENET_MEAN if mean is None else mean
        self.std = self.IMAGENET_STD if std is None else std

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
        return self.mean_std_normalize(img)

if __name__=="__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    from pyzjr.augmentation.transforms.applytransformer import Compose
    matplotlib.use("TkAgg")

    image_path = r'../../data/scripts/fire.jpg'
    cv_image = cv2.imread(image_path)
    pil_image = Image.open(image_path)

    transforms = Compose([
        # RandomHorizontalFlip(),
        # RandomVerticalFlip(),
        # AdjustBrightness(0.8),
        # RandomAdjustBrightness(1.2),
        # AdjustGamma(1.2),
        # RandomAdjustGamma(1.3),
        # ToHSV(),
        # RandomRotation(),
        # GaussianBlur(ksize=3),
        # Grayscale(),
        # EqualizeHistogram(),
        # RandomEqualizeHistogram(),
        # Centerzoom(1.6),
        # Resize(target_size=(256, 256)),
        # RandomCrop(target_size=128),
        # CenterCrop(target_size=128),
        # RandomResizeCrop(target_size=128),
        # Pad((40, 15)),
        # ResizePad(256),
        # InvertColor(),
        # RandomInvertColor(),
        # AdjustSharpness(1.2),
        # RandomAdjustSharpness(1.5),
        # AdjustSaturation(3),
        # RandomAdjustSaturation(3),
        # ColorJitter(),
        # Normalizer('zero_centered'),
        ToTensor(),
        MeanStdNormalize()
    ])
    transformed_pil_image = transforms(pil_image)
    transformed_cv_image = transforms(cv_image)

    print(transformed_pil_image.shape)
    print(transformed_cv_image.shape)
    transformed_pil_image = torch.clamp(transformed_pil_image, 0, 1)
    transformed_pil_image = transformed_pil_image.permute(1, 2, 0)

    transformed_cv_image = torch.clamp(transformed_cv_image, 0, 1)
    transformed_cv_image = transformed_cv_image.permute(1, 2, 0)

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(transformed_pil_image)
    plt.title('Transformed PIL Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(transformed_cv_image)
    plt.title('Transformed CV Image')
    plt.axis('off')

    plt.show()
