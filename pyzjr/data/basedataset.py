import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
from pyzjr.core import to_2tuple
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """A simple dataset class with the same functionality as torch.utils.data.Dataset"""
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def assert_list_length(self, image, label):
        assert (len(image) == len(label)), r"The number of loaded images and labels does not match"

    def list_files(self, imagefolder, image_extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        """
        返回给定目录中的所有图像文件
        """
        return [
            f for f in Path(imagefolder).iterdir()
            if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in image_extensions
        ]

    def list_dirs(self, imagefolder):
        """
        返回给定目录中的所有目录
        """
        return [f for f in Path(imagefolder).iterdir() if f.is_dir()]

    def rand(self, a=0., b=1.):
        """
        生成范围在 [a, b) 之间的随机数
        """
        return np.random.rand() * (b - a) + a

    def osmkdirs(self, *args):
        """
        如果不存在的话,创建多个目录
        """
        for path in args:
            if not os.path.exists(path):
                os.makedirs(path)

    def preprocess_input(self, image):
        """
        将图像转换为浮点数并进行归一化
        """
        if np.max(np.array(image)) > 1:
            normalized_image = np.asarray(image) / 255.0
            return normalized_image.astype(np.float64)
        else:
            return image

    def Shuffle(self, files):
        """
        随机打乱文件列表
        """
        assert isinstance(files, (list, tuple)), "Input must be a list or tuple"
        random.shuffle(files)
        return files

    def load_txt_path_or_list(self, txt_path):
        """
        加载txt文件或列表,按行读取,去掉'/n'
        """
        path_list = []
        if isinstance(txt_path, str):  # Check if txt_path is a string (path to a file)
            with open(txt_path, 'r') as file:
                lines = file.readlines()
                path_list.extend(line.strip() for line in lines)
        elif isinstance(txt_path, list):  # Check if txt_path is a list
            path_list.extend(item.strip() for item in txt_path)
        else:
            raise ValueError(f"Invalid input type. {txt_path} should be a string (file path) or a list.")
        return path_list

    def resizepadcenter(self, image, label, input_shape):
        """
        将图像和标签粘贴到指定输入形状的中心位置。

        :param image: 输入图像 (PIL.Image)
        :param label: 输入标签图像 (PIL.Image)
        :param input_shape: 目标输入形状 (tuple)，形如 (width, height)
        :return: 调整后的图像和标签
        """
        input_shape = to_2tuple(input_shape)
        h, w = input_shape
        iw, ih = image.size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

        label = label.resize((nw, nh), Image.NEAREST)
        new_label = Image.new('RGB', (w, h), (0, 0, 0))
        new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
        return new_image, new_label