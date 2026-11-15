"""
Copyright (c) 2022, Auorui.
All rights reserved.
"""
import cv2
import math
import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as tf

from pyzjr.aug.geometric import smart_resize, resize_pad


def is_pil(param):
    return isinstance(param, Image.Image)


def is_numpy(param):
    """Check if the input image is Numpy format."""
    return isinstance(param, np.ndarray)


def is_pil_or_tensor(param):
    return isinstance(param, (Image.Image, torch.Tensor))


class RandomFlip:
    def __init__(self, p=0.5, mode=1):
        self.p = p
        self.mode = mode   # 1=horizontal, 0=vertical, -1=both

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target

        if is_pil_or_tensor(image) and is_pil_or_tensor(target):
            if self.mode == 1:
                image = tf.hflip(image)
                target = tf.hflip(target)
            elif self.mode == 0:
                image = tf.vflip(image)
                target = tf.vflip(target)
            else:  # mode == -1 (both)
                image = tf.vflip(tf.hflip(image))
                target = tf.vflip(tf.hflip(target))

        elif is_numpy(image) and is_numpy(target):
            image = cv2.flip(image, self.mode)
            target = cv2.flip(target, self.mode)

        return image, target


class ResizePad:
    def __init__(self, size, fill=128, label_fill=0):
        """
        Resize the image and target while maintaining aspect ratio, then pad to the target size.
        (Commonly known as Letterbox padding)

        Args:
            size: Target size. If int, it will be squared (size, size).
                  If tuple/list, it should be (width, height).
            fill (int/tuple): Fill value for the image (default: 128).
            label_fill (int/tuple): Fill value for the target mask (default: 0).
        """
        if isinstance(size, int):
            self.size = [size, size]  # [w, h]
        else:
            self.size = size
        self.fill = fill
        self.label_fill = label_fill
        self.pad_meta = None

    def __call__(self, image, target):
        tw, th = self.size[0], self.size[1]
        if is_pil_or_tensor(image) and is_pil_or_tensor(target):
            _, h, w = tf.get_dimensions(image)
            scale = min(tw / w, th / h)
            nw, nh = int(w * scale), int(h * scale)

            pad_w = tw - nw
            pad_h = th - nh

            left = pad_w // 2
            top = pad_h // 2
            right = pad_w - left
            bottom = pad_h - top
            padding = [left, top, right, bottom]

            self.pad_meta = (top, left, nh, nw, h, w)

            image = tf.resize(image, [nh, nw], interpolation=tf.InterpolationMode.BILINEAR)
            image = tf.pad(image, padding, fill=self.fill, padding_mode='constant')

            target = tf.resize(target, [nh, nw], interpolation=tf.InterpolationMode.NEAREST)
            target = tf.pad(target, padding, fill=self.label_fill, padding_mode='constant')

        elif is_numpy(image) and is_numpy(target):
            image, self.pad_meta = resize_pad(image, self.size, (self.fill, self.fill, self.fill), is_label=False)
            target, _ = resize_pad(target, self.size, (self.label_fill, self.label_fill, self.label_fill), is_label=True)

        return image, target




class CenterCrop:
    def __init__(self, size):
        """
        Args:
            size: Target size. If int, it will be squared (size, size).
                  If tuple/list, it should be (width, height)
        """
        if isinstance(size, int):
            self.size = [size, size]
        else:
            self.size = size

    def __call__(self, image, target):
        if is_pil_or_tensor(image) and is_pil_or_tensor(target):
            image = tf.center_crop(image, self.size)
            target = tf.center_crop(target, self.size)

        elif is_numpy(image) and is_numpy(target):
            h, w = image.shape[:2]
            crop_w, crop_h = self.size

            top = (h - crop_h) // 2
            left = (w - crop_w) // 2
            bottom = top + crop_h
            right = left + crop_w

            image = image[top:bottom, left:right]
            target = target[top:bottom, left:right]

        return image, target


class RandomCrop:
    def __init__(self, size):
        """
        Args:
            size: Target size. If int, it will be squared (size, size).
                  If tuple/list, it should be (width, height) or (height, width).
        """
        if isinstance(size, int):
            self.size = [size, size]
        else:
            self.size = size

    def __call__(self, image, target):
        if is_pil_or_tensor(image) and is_pil_or_tensor(target):
            crop_w, crop_h = self.size
            _, h, w = tf.get_dimensions(image)
            if crop_h > h or crop_w > w:
                raise ValueError(f"Crop size {(crop_h, crop_w)} is larger than image size {(h, w)}.")

            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)

            image = tf.crop(image, top, left, crop_h, crop_w)
            target = tf.crop(target, top, left, crop_h, crop_w)

        elif is_numpy(image) and is_numpy(target):
            h, w = image.shape[:2]
            crop_h, crop_w = self.size

            if crop_h > h or crop_w > w:
                raise ValueError(f"Crop size {(crop_h, crop_w)} is larger than image size {(h, w)}.")

            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            bottom = top + crop_h
            right = left + crop_w

            image = image[top:bottom, left:right]
            target = target[top:bottom, left:right]

        return image, target


class Resize:
    def __init__(self, size):
        """
        Args:
            size: Target size. If int, it will be squared (size, size).
                  If tuple/list, it should be (width, height).
        """
        if isinstance(size, int):
            self.size = [size, size]  # [w, h]
        else:
            self.size = size

    def __call__(self, image, target):
        if is_pil_or_tensor(image) and is_pil_or_tensor(target):
            image = tf.resize(
                image,
                size=[self.size[1], self.size[0]],
                interpolation=tf.InterpolationMode.BILINEAR
            )
            target = tf.resize(
                target,
                size=[self.size[1], self.size[0]],
                interpolation=tf.InterpolationMode.NEAREST
            )

        elif is_numpy(image) and is_numpy(target):
            image = smart_resize(image, self.size, is_label=False)
            target = smart_resize(target, self.size, is_label=True)

        return image, target


class RandomResizedCrop:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
        self.size = [size, size] if isinstance(size, int) else size
        self.scale = scale
        self.ratio = ratio

    def _get_crop_box(self, h, w):
        area = h * w
        target_area = random.uniform(*self.scale) * area

        aspect_ratio = math.exp(
            random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))

        crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_h = int(round(math.sqrt(target_area / aspect_ratio)))

        if crop_w > w:
            crop_w = w
            crop_h = int(round(crop_w / aspect_ratio))
        if crop_h > h:
            crop_h = h
            crop_w = int(round(crop_h * aspect_ratio))

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        return top, left, crop_h, crop_w

    def __call__(self, image, target):
        if is_pil_or_tensor(image) and is_pil_or_tensor(target):
            _, h, w = tf.get_dimensions(image)
            top, left, crop_h, crop_w = self._get_crop_box(h, w)

            image = tf.resized_crop(
                image, top, left, crop_h, crop_w, self.size,
                interpolation=tf.InterpolationMode.BILINEAR
            )
            target = tf.resized_crop(
                target, top, left, crop_h, crop_w, self.size,
                interpolation=tf.InterpolationMode.NEAREST
            )

        elif is_numpy(image) and is_numpy(target):
            h, w = image.shape[:2]
            top, left, crop_h, crop_w = self._get_crop_box(h, w)

            image = image[top:top+crop_h, left:left+crop_w]
            target = target[top:top+crop_h, left:left+crop_w]

            image = smart_resize(image, self.size, is_label=False)
            target = smart_resize(target, self.size, is_label=True)

        return image, target


class RandomRotation90:
    """
    Randomly rotate the image and target by 0, 90, 180, or 270 degrees.
    Assumes the input image is square to avoid boundary padding.
    Supports PIL, Tensor, and Numpy formats.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target

        k = random.randint(0, 3)
        if k == 0:
            return image, target

        if is_pil_or_tensor(image) and is_pil_or_tensor(target):
            angle = k * 90
            image = tf.rotate(
                image, angle=angle, interpolation=tf.InterpolationMode.BILINEAR, expand=False
            )
            target = tf.rotate(
                target, angle=angle, interpolation=tf.InterpolationMode.NEAREST, expand=False
            )

        elif is_numpy(image) and is_numpy(target):
            image = np.rot90(image, k).copy()
            target = np.rot90(target, k).copy()

        return image, target


class RandomBrightness:
    """Randomly adjust the brightness of the image. (Target remains unchanged)"""
    def __init__(self, brightness=0.2):
        self.brightness = brightness

    def __call__(self, image, target):
        if self.brightness <= 0:
            return image, target

        factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)

        if is_pil_or_tensor(image) and is_pil_or_tensor(target):
            image = tf.adjust_brightness(image, factor)
        elif is_numpy(image) and is_numpy(target):
            image = cv2.convertScaleAbs(image, alpha=factor, beta=0)

        return image, target


class RandomContrast:
    """Randomly adjust the contrast of the image."""
    def __init__(self, contrast=0.2):
        self.contrast = contrast

    def __call__(self, image, target):
        if self.contrast <= 0:
            return image, target

        factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)

        if is_pil_or_tensor(image) and is_pil_or_tensor(target):
            image = tf.adjust_contrast(image, factor)
        elif is_numpy(image) and is_numpy(target):
            # Contrast Formula:(img - mean) * factor + mean
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            image = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

        return image, target


class RandomSaturation:
    """Randomly adjust the saturation of the image."""
    def __init__(self, saturation=0.2):
        self.saturation = saturation

    def __call__(self, image, target):
        if self.saturation <= 0:
            return image, target

        factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)

        if is_pil_or_tensor(image) and is_pil_or_tensor(target):
            image = tf.adjust_saturation(image, factor)
        elif is_numpy(image) and is_numpy(target):
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return image, target


class RandomHue:
    """Randomly adjust the hue of the image."""
    def __init__(self, hue=0.1):
        self.hue = hue

    def __call__(self, image, target):
        if self.hue <= 0:
            return image, target

        factor = random.uniform(-self.hue, self.hue)

        if is_pil_or_tensor(image) and is_pil_or_tensor(target):
            image = tf.adjust_hue(image, factor)
        elif is_numpy(image) and is_numpy(target):
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            delta_h = factor * 180
            hsv[:, :, 0] = (hsv[:, :, 0] + delta_h) % 180
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return image, target


class ColorJitter:
    """
    Randomly change the brightness, contrast, saturation and hue of an image.
    Supports PIL, Tensor, and Numpy formats.
    """
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target

        brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness) if self.brightness > 0 else None
        contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast) if self.contrast > 0 else None
        saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation) if self.saturation > 0 else None
        hue_factor = random.uniform(-self.hue, self.hue) if self.hue > 0 else None

        if is_pil_or_tensor(image) and is_pil_or_tensor(target):
            if brightness_factor is not None:
                image = tf.adjust_brightness(image, brightness_factor)
            if contrast_factor is not None:
                image = tf.adjust_contrast(image, contrast_factor)
            if saturation_factor is not None:
                image = tf.adjust_saturation(image, saturation_factor)
            if hue_factor is not None:
                image = tf.adjust_hue(image, hue_factor)

        elif is_numpy(image) and is_numpy(target):
            if brightness_factor is not None:
                image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

            if contrast_factor is not None:
                mean = np.mean(image, axis=(0, 1), keepdims=True)
                image = np.clip((image.astype(np.float32) - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)

            if saturation_factor is not None or hue_factor is not None:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
                h, s, v = cv2.split(hsv)

                if saturation_factor is not None:
                    s = np.clip(s * saturation_factor, 0, 255)

                if hue_factor is not None:
                    delta_h = hue_factor * 180
                    h = (h + delta_h) % 180

                image = cv2.cvtColor(cv2.merge([h, s, v]).astype(np.uint8), cv2.COLOR_HSV2RGB)
        return image, target


class ToTensor(object):
    """
    Convert a PIL Image or numpy.ndarray to a PyTorch Tensor.
    - Image: Converts to float32 in range [0.0, 1.0], shape (C, H, W).
    - Target: Converts to int64 (Long), shape (C, H, W) for RGB masks.
    - If the input is already a Tensor, it will be returned directly.
    """
    def __init__(self):
        super(ToTensor, self).__init__()

    def __call__(self, image, target=None):
        if is_pil(image) or is_numpy(image):
            image = tf.to_tensor(image)   # hwc -> chw / 255
        elif torch.is_tensor(image):
            pass
        else:
            raise TypeError(f"image should be PIL Image, Tensor or ndarray. Got {type(image)}")

        if target is not None:
            if is_pil_or_tensor(target):
                target = np.array(target)

            if is_numpy(target):
                # Label: Keep three channels! Do not divide by 255, only convert to Long
                # Direct image mapping is performed, so normalization is not required
                if target.ndim == 2:
                    target = target[:, :, None]
                target = torch.from_numpy(target.transpose((2, 0, 1))).contiguous()
                target = target.long()
            else:
                raise TypeError(f"target should be PIL Image or ndarray. Got {type(target)}")

            return image, target
        else:
            return image


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = (0.485, 0.456, 0.406) if mean is None else mean
        self.std = (0.229, 0.224, 0.225) if std is None else std

    def __call__(self, image, target):
        image = tf.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ToPILImage(object):
    """
    Convert a Tensor or an ndarray to PIL Image.
    - If Tensor is [0.0, 1.0], it will be scaled to [0, 255] and cast to uint8.
    - If Tensor is [0, 255] or int, it will just be cast to uint8.
    """
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, image):
        if is_numpy(image):
            if image.dtype == np.float32 or image.max() <= 1.0:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
            image = Image.fromarray(image)
        elif torch.is_tensor(image):
            image = tf.to_pil_image(image)
        elif is_pil(image):
            pass
        else:
            raise TypeError(f"Unsupported type: {type(image)}")
        if self.mode is not None:
            image = image.convert(self.mode)
        return image


class ToNdarrayImage(object):
    """
    Convert a Tensor or PIL Image to a numpy.ndarray.
    - Converts CHW to HWC.
    - If Tensor is [0.0, 1.0], it will be scaled to [0, 255] and cast to uint8.
    """
    def __init__(self, to_bgr=False):
        self.to_bgr = to_bgr

    def __call__(self, image):
        if torch.is_tensor(image):
            image = image.detach().cpu()
            if image.dtype in (torch.float16, torch.float32) and image.max() <= 1.0:
                image = (image * 255).clip(0, 255).byte().numpy()
            else:
                image = image.numpy()
            # CHW -> HWC
            if image.ndim == 3:
                image = np.transpose(image, (1, 2, 0))

        elif is_pil(image):
            image = np.array(image)
        elif is_numpy(image):
            pass
        else:
            raise TypeError(f"Unsupported type: {type(image)}")

        if self.to_bgr and image.ndim == 3 and image.shape[-1] == 3:
            image = image[:, :, ::-1].copy()

        return image


class Compose(object):
    """
    Compose multiple transforms together. (Equivalent to torchvision.transforms.Compose)

    This class sequentially applies a list of transforms to the image and target.
    It automatically handles cases where a transform only expects the image (no target).
    """
    def __init__(self, transforms: list):
        """
        Args:
            transforms (list): A list of transform callables. Each callable should accept
                               (image, target) or just (image).
        """
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            try:
                res = t(image, target)
                if isinstance(res, tuple) and len(res) == 2:
                    image, target = res
                else:
                    image = res
            except TypeError:
                image = t(image)

        if target is not None:
            return image, target
        return image

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from pyzjr.io import matplotlib_patch
    matplotlib_patch()
    image_path = r'E:\PythonProject\pyzjrPyPi\resources/images/aircraft.jpg'
    label_path = r'E:\PythonProject\pyzjrPyPi\resources/images/aircraft_mask.png'

    img_pil = Image.open(image_path).convert('RGB')
    mask_pil = Image.open(label_path).convert('RGB')

    img_np = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    mask_np = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB)

    train_transform = Compose([
        RandomFlip(p=0.5, mode=1),               # 随机水平翻转
        # RandomCrop(size=256),                    # 随机裁剪 256x256
        # CenterCrop(size=256),
        ResizePad(size=256),
        # RandomResizedCrop(size=256),
        # Resize(size=256),
        RandomRotation90(p=0.8),                 # 随机 90 度旋转（需保证正方形输入，所以放在 Resize 后）
        # RandomBrightness(.2),
        # RandomContrast(.2),
        # RandomSaturation(.2),
        # RandomHue(.2),
        ColorJitter(brightness=0.2, p=0.5),      # 颜色抖动
        ToTensor(),                              # 转 Tensor (Float, [0,1], CHW)
        # Normalize()                              # ImageNet 标准化
    ])

    out_img_pil, out_mask_pil = train_transform(img_pil, mask_pil)
    print(f"PIL Image Tensor: shape={out_img_pil.shape}, dtype={out_img_pil.dtype}, range=[{out_img_pil.min():.2f}, {out_img_pil.max():.2f}]")
    print(f"PIL Mask  Tensor: shape={out_mask_pil.shape}, dtype={out_mask_pil.dtype}, unique={torch.unique(out_mask_pil).tolist()}")

    out_img_np, out_mask_np = train_transform(img_np, mask_np)
    print(f"Ndarray Image Tensor: shape={out_img_np.shape}, dtype={out_img_np.dtype}")
    print(f"Ndarray Mask  Tensor: shape={out_mask_np.shape}, dtype={out_mask_np.dtype}")

    # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    # raw_img_tensor = out_img_pil * std + mean  # 反标准化

    # 用你的 ToNdarrayImage 转换，并设置 to_bgr=True 方便 cv2 显示
    to_numpy = ToNdarrayImage(to_bgr=True)
    display_img = to_numpy(out_img_pil)
    display_mask = to_numpy(out_mask_pil)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Transformed Image (BGR -> RGB)")
    plt.imshow(display_img[:, :, ::-1])
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Transformed Mask (RGB)")
    plt.imshow(display_mask)
    plt.axis('off')

    plt.tight_layout()
    plt.show()