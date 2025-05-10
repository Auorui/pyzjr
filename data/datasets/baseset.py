"""
Copyright (c) 2024, Auorui.
All rights reserved.

This module is used as the base class for homemade data loaders.
"""
import cv2
import os
import random
import numpy as np
from torch.utils.data import Dataset
from pyzjr.utils.check import is_tensor, is_numpy

class BaseDataset(Dataset):
    """A simple dataset class with the same functionality as torch.utils.data.Dataset"""
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def read_image(self, filename, target_shape=None, to_rgb=True, normalize=True):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {filename}")
        if target_shape is not None:
            img = self.resizepad(img, target_shape)
        img = img[:, :, ::-1] if to_rgb else img
        return self.preprocess_input(img) if normalize else img

    def resizepad(self, image, target_shape, label=None):
        h, w = target_shape
        ih, iw = image.shape[:2]
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        new_image = np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)
        top = (h - nh) // 2
        left = (w - nw) // 2
        new_image[top:top + nh, left:left + nw] = resized_image
        if label is not None:
            resized_label = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST)
            new_label = np.zeros((h, w), dtype=np.uint8)
            new_label[top:top + nh, left:left + nw] = resized_label
            return new_image, new_label
        else:
            return new_image

    def colormap2label(self, image, colormap, dtype="int64"):
        colormap_label = np.zeros(256 ** 3, dtype=np.int64)
        image = np.array(image, dtype=dtype)
        for i, color_map in enumerate(colormap):
            colormap_label[(color_map[0] * 256 + color_map[1]) * 256 + color_map[2]] = i
        idx = ((image[:, :, 0] * 256 + image[:, :, 1]) * 256
               + image[:, :, 2])
        return colormap_label[idx]

    def rand(self, a=0., b=1.):
        return random.uniform(a, b)

    def shuffle(self, x):
        random.seed(0)
        return random.sample(x, len(x))

    def multi_makedirs(self, *args):
        for path in args:
            if not os.path.exists(path):
                os.makedirs(path)

    def preprocess_input(self, image, mean=None, std=None):
        image = np.asarray(image)
        if np.max(image) > 1:
            image = image / 255.0

        if mean is not None:
            mean = np.array(mean)
            image = image - mean

        if std is not None:
            std = np.array(std)
            image = image / std

        return image.astype(np.float32)

    def read_txt(self, txt_path):
        path_list = []
        if isinstance(txt_path, str):  # Check if txt_path is a string
            with open(txt_path, 'r') as file:
                lines = file.readlines()
                path_list.extend(line.strip() for line in lines)
        return path_list

    def hwc2chw(self, image):
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

    def auguments(self, imglists, target_shape=None, prob=.5):
        if target_shape is not None:
            H, W, _ = imglists[0].shape
            Hc, Wc = target_shape
            Hs = random.randint(0, H - Hc)
            Ws = random.randint(0, W - Wc)
            for i in range(len(imglists)):
                imglists[i] = imglists[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

        # horizontal flip
        if random.random() > (1 - prob):
            for i in range(len(imglists)):
                imglists[i] = np.flip(imglists[i], axis=1).copy()

        r = random.randint(0, 3)
        for i in range(len(imglists)):
            imglists[i] = np.rot90(imglists[i], r, (0, 1)).copy()

        return imglists

    def align(self, imglists, target_shape):
        H, W, _ = imglists[0].shape
        Hc, Wc = target_shape
        Hs = (H - Hc) // 2
        Ws = (W - Wc) // 2
        for i in range(len(imglists)):
            imglists[i] = imglists[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

        return imglists

    def disable_cv2_multithreading(self):
        """
        Disable OpenCV's multithreading and OpenCL usage for consistent behavior and performance.
        """
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

if __name__=="__main__":
    datasets = BaseDataset()
    x = ['1', '2', '3', '4', '5']
    x = datasets.shuffle(x)
    print(x)