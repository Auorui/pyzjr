import os
import random
import numpy as np
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """A simple dataset class with the same functionality as torch.utils.data.Dataset"""
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def rand(self, a=0., b=1.):
        """
        生成范围在 [a, b) 之间的随机数
        """
        return random.uniform(a, b)

    def osmkdirs(self, *args):
        """
        如果不存在的话, 创建多个目录
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

    def read_txt(self, txt_path):
        path_list = []
        if isinstance(txt_path, str):  # Check if txt_path is a string
            with open(txt_path, 'r') as file:
                lines = file.readlines()
                path_list.extend(line.strip() for line in lines)
        return path_list
