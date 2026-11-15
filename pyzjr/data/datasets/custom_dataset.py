"""
Copyright (c) 2024, Auorui.
All rights reserved.

Enhanced base dataset class with comprehensive image processing utilities.
"""
import re
import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Callable
from pyzjr.data.core.path import list_files, IMG_EXTENSIONS
from pyzjr.aug.geometric import random_flip, random_crop, center_crop, random_rot90, resize_pad_lists


class BaseDataset(Dataset):
    """A simple dataset class with the same functionality as torch.utils.data.Dataset"""
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    # -------------------- Core Image IO --------------------
    def read_image(self, filename: str, to_rgb: bool, normalize: bool,
                   transforms: Optional[List[Callable]] = None) -> np.ndarray:
        if not os.path.exists(filename):
            raise ValueError(f"The file {filename} does not exist.")
        if not isinstance(filename, str):
            raise TypeError("image_path must be a string")
        if transforms is not None and not all(callable(t) for t in transforms):
            raise ValueError("transforms must be a list of callable functions")
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {filename}")
        if transforms:
            for transform in transforms:
                image = transform(image)
        if to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        if normalize:
            image /= 255
        return image

    def disable_cv2_multithreading(self):
        """
        Disable OpenCV's multithreading and OpenCL usage for consistent behavior and performance.
        """
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

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

    def get_file_path(self, dir_path, suffix=IMG_EXTENSIONS, return_abs_path=False):
        """Search for files in a directory and return a list of file paths."""
        return list_files(dir_path, suffix=suffix, return_abs_path=return_abs_path)

    def natsorted(self, a):
        """Perform natural sorting of strings containing numbers."""
        def _natural_key(st):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split('([0-9]+)', st)]
        return sorted(a, key=_natural_key)

    # -------------------- Image Format Conversion --------------------
    def hwc2chw(self, image):
        """Convert HWC format to CHW format."""
        if isinstance(image, Image.Image):  # PIL.Image 支持
            image = np.array(image)
        if len(image.shape) == 3:
            if isinstance(image, np.ndarray):
                return np.transpose(image, axes=[2, 0, 1]).copy()
            elif torch.is_tensor(image):
                return image.permute(2, 0, 1).contiguous()
            else:
                raise TypeError("The input data should be a NumPy array or "
                                "PyTorch tensor, but the provided type is: {}".format(type(image)))
        else:
            raise ValueError("The input data should be three-dimensional (height x width x channel), but the "
                             "provided number of dimensions is:{}".format(len(image.shape)))

    # -------------------- Advanced Augmentations --------------------
    def augment(self,
                imglists: List[np.ndarray],
                target_shape: Optional[Tuple[int, int]],
                is_labels,
                prob: float = 0.5,
                transforms=None) -> List[np.ndarray]:
        """Apply random augmentations including crop, flip, rotate, and color jitter.

        Args:
            imglists: List of images to augment (HWC format).
            target_shape: (width, height) for random crop.
            prob: Probability of applying horizontal flip.
            transforms: List of callable functions to apply sequentially.
            is_labels (bool or list, optional): Whether the input is a label map.
                Labels use nearest-neighbor interpolation during resizing. Defaults to None.

        Returns:
            List of augmented images.
        """
        if target_shape is not None:
            imglists = random_crop(imglists, target_shape, is_labels=is_labels)

        # horizontal flip
        imglists = random_flip(imglists, axis=1, prob=prob)

        # Random rotation (0°, 90°, 180°, 270°)
        imglists = random_rot90(imglists)

        if transforms:
            for transform in transforms:
                imglists = [transform(img) for img in imglists]

        return imglists

    def align(self, imglists, target_shape, is_labels, mode="center_crop", transforms=None):
        """Central clipping, used during the validation phase."""
        if target_shape is not None:
            if mode == 'center_crop':
                imglists = center_crop(imglists, target_shape, is_labels=is_labels)
            elif mode == 'resize_pad':
                imglists = resize_pad_lists(imglists, target_shape,
                                            pad_color=(128, 128, 128),
                                            is_labels=is_labels)
            else:
                raise ValueError(f"mode must be 'center_crop' or 'resize_pad', got {mode!r}")
        if transforms:
            for transform in transforms:
                imglists = [transform(img) for img in imglists]

        return imglists
