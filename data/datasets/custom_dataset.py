"""
Copyright (c) 2024, Auorui.
All rights reserved.

Enhanced base dataset class with comprehensive image processing utilities.
"""
import re
import os
import cv2
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pyzjr.utils.check import is_tensor, is_numpy
from typing import List, Optional, Tuple, Union, Callable

class BaseDataset(Dataset):
    """A simple dataset class with the same functionality as torch.utils.data.Dataset"""
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    # -------------------- Core Image IO --------------------
    def read_image(self, image_path: str, to_rgb: bool, normalize: bool,
                   transformers: Optional[List[Callable]] = None) -> np.ndarray:
        """Read an image with optional transformations.

        Args:
            image_path: Path to the image file.
            to_rgb: Convert BGR to RGB if True.
            normalize: Scale pixel values to [0, 1] if True.
            transformers: List of callable functions to apply sequentially.

        Returns:
            float32 numpy array of shape (H, W, C).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        if not isinstance(image_path, str):
            raise TypeError("image_path must be a string")
        if transformers is not None and not all(callable(t) for t in transformers):
            raise ValueError("transformers must be a list of callable functions")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        if transformers:
            for transformer in transformers:
                image = transformer(image)
        if to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        if normalize:
            image /= 255
        return image

    def preprocess_input(self, image):
        image = image.astype(np.float32)
        return image / 255.0

    # -------------------- Basic Utilities --------------------
    def rand(self, a: float = 0., b: float = 1.) -> float:
        """Generate random float in [a, b]."""
        return random.uniform(a, b)

    def to_2tuple(self, x):
        """Convert input to a 2-element tuple"""
        if isinstance(x, int):
            return (x, x)
        elif isinstance(x, (list, tuple)):
            if len(x) < 2:
                raise ValueError(f"Input sequence must contain at least 2 elements, got {len(x)}")
            return (x[0], x[1])
        raise TypeError(f"Input must be int or sequence, got {type(x)}")

    def shuffle(self, x: list, seed: int = 0) -> list:
        """Shuffle a list with optional seed."""
        random.seed(seed)
        return random.sample(x, len(x))

    def multi_makedirs(self, *paths: str):
        """Create multiple directories if they don't exist."""
        for path in paths:
            os.makedirs(path, exist_ok=True)

    def read_txt(self, txt_path: str) -> List[str]:
        """Read lines from a text file into a list."""
        with open(txt_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def SearchFilePath(self, target_path, file_exts=('.png', '.jpg', '.jpeg', '.bmp')):
        """Search for files with specified extensions in the target directory (non-recursive)."""
        search_file_path = []
        files = os.listdir(target_path)
        for filespath in files:
            if str(filespath).endswith(file_exts):
                search_file_path.append(os.path.join(target_path, filespath))
        return self.natsorted(search_file_path)

    def SearchFileName(self, target_path, file_exts=('.png', '.jpg', '.jpeg', '.bmp')):
        """Get filenames with specified extensions in the target directory (non-recursive)."""
        all_files = os.listdir(target_path)
        image_files = [file for file in all_files if file.lower().endswith(file_exts)]
        return self.natsorted(image_files)

    def natsorted(self, a):
        """Perform natural sorting of strings containing numbers."""
        def natural_key(st):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split('([0-9]+)', st)]
        return sorted(a, key=natural_key)

    # -------------------- Image Format Conversion --------------------
    def hwc2chw(self, image):
        """Convert HWC format to CHW format."""
        if isinstance(image, Image.Image):  # PIL.Image 支持
            image = np.array(image)
        if len(image.shape) == 3:
            if is_numpy(image):
                return np.transpose(image, axes=[2, 0, 1])
            elif is_tensor(image):
                return image.permute(2, 0, 1).contiguous()
            else:
                raise TypeError("The input data should be a NumPy array or "
                                "PyTorch tensor, but the provided type is: {}".format(type(image)))
        else:
            raise ValueError("The input data should be three-dimensional (height x width x channel), but the "
                             "provided number of dimensions is:{}".format(len(image.shape)))

    # -------------------- OpenCV Configuration --------------------
    def disable_cv2_multithreading(self):
        """
        Disable OpenCV's multithreading and OpenCL usage for consistent behavior and performance.
        """
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

    # -------------------- Advanced Augmentations --------------------
    def augment(self,
                imglists: List[np.ndarray],
                target_shape: Optional[Tuple[int, int]] = None,
                prob: float = 0.5,
                transforms=None) -> List[np.ndarray]:
        """Apply random augmentations including crop, flip, rotate, and color jitter.

        Args:
            imglists: List of images to augment (HWC format).
            target_shape: (width, height) for random crop.
            prob: Probability of applying horizontal flip.
            transforms: List of callable functions to apply sequentially.

        Returns:
            List of augmented images.
        """
        if target_shape is not None:
            H, W, _ = imglists[0].shape
            Wc, Hc = target_shape
            Hs = random.randint(0, H - Hc)
            Ws = random.randint(0, W - Wc)
            for i in range(len(imglists)):
                imglists[i] = imglists[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

        # horizontal flip
        if random.random() < prob:
            imglists = [np.flip(img, axis=1).copy() for img in imglists]

        # Random rotation (0°, 90°, 180°, 270°)
        k = random.randint(0, 3)
        imglists = [np.rot90(img, k, (0, 1)).copy() for img in imglists]

        if transforms:
            for transform in transforms:
                imglists = [transform(img) for img in imglists]

        return imglists

    def align(self, imglists, target_shape, transforms=None):
        """Central clipping, used during the validation phase"""
        H, W, _ = imglists[0].shape
        Wc, Hc = target_shape
        Hs = (H - Hc) // 2
        Ws = (W - Wc) // 2
        for i in range(len(imglists)):
            imglists[i] = imglists[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]
        if transforms:
            for transform in transforms:
                imglists = [transform(img) for img in imglists]
        return imglists

    def color_jitter(self, image: np.ndarray, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2, hue: float = 0.1) -> np.ndarray:
        """Apply random color jitter to an image."""
        if image.dtype == np.float32 and np.max(image) <= 1.0:
            raise ValueError("Input image appears to be already normalized (values in [0,1]). "
                             "Please provide image in [0,255] range.")
        image = image.astype(np.float32) / 255.0
        # Brightness adjustment (additive)
        if brightness > 0:
            delta = np.random.uniform(-brightness, brightness)
            image = np.clip(image + delta, 0, 1)

        # Contrast adjustment (multiplicative)
        if contrast > 0:
            alpha = 1.0 + np.random.uniform(-contrast, contrast)
            image = np.clip(alpha * image, 0, 1)

        # Saturation & Hue adjustment (HSV space)
        if saturation > 0 or hue > 0:
            # Convert to HSV (OpenCV expects float32 in [0, 1] for H, [0, 255] for S/V)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)

            # Saturation scaling
            if saturation > 0:
                alpha = 1.0 + np.random.uniform(-saturation, saturation)
                s = np.clip(alpha * s, 0, 1)

            # Hue shifting (OpenCV Hue range: [0, 180] for uint8)
            if hue > 0:
                delta = np.random.uniform(-hue, hue) * 180  # Convert radians to OpenCV Hue units
                h = (h * 180 + delta) % 180  # Scale float H to [0,180], then shift
                h = h / 180  # Scale back to [0,1]

            image = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)
        image = (image * 255).clip(0, 255).astype(np.uint8)
        return image

    def resizepad(self, image, target_shape, label=None, pad_color=(128, 128, 128)):
        """Resize image with aspect ratio kept and pad to target shape."""
        w, h = target_shape
        ih, iw = image.shape[:2]
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        is_normalized = image.dtype == np.float32 and np.max(image) <= 1.0
        if is_normalized:
            pad_color = tuple(c/255.0 for c in pad_color)
            output_dtype = np.float32
        else:
            output_dtype = np.uint8
        interp = cv2.INTER_LINEAR if scale < 1 else cv2.INTER_CUBIC
        resized_image = cv2.resize(image, (nw, nh), interpolation=interp)
        new_image = np.full((h, w, 3), pad_color, dtype=output_dtype)
        top = (h - nh) // 2
        left = (w - nw) // 2
        new_image[top:top + nh, left:left + nw] = resized_image
        if label is not None:
            resized_label = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST)
            if label.ndim == 3:
                new_label = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                new_label = np.zeros((h, w), dtype=np.uint8)
            new_label[top:top + nh, left:left + nw] = resized_label
            return new_image, new_label
        else:
            return new_image

    def random_perspective(self, image: np.ndarray, degrees: float = 10, translate: float = 0.1, scale: float = 0.1) -> np.ndarray:
        """Apply random perspective transformation."""
        height, width = image.shape[:2]
        # Rotation and scale
        angle = self.rand(-degrees, degrees)
        scale_val = self.rand(1 - scale, 1 + scale)
        # Translation
        tx = translate * width * self.rand(-1, 1)
        ty = translate * height * self.rand(-1, 1)
        # Transformation matrix
        M = cv2.getRotationMatrix2D((width/2, height/2), angle, scale_val)
        M[:, 2] += (tx, ty)
        return cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
