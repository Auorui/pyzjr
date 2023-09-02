import os
import torchvision
import torch
import pyzjr.Z as Z
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


def voc_catalog():
    print('VOC Catalog:')
    print('-' * 70)
    print("VOCdevkit")
    print("    VOC2007")
    print("        -ImageSets/Segmentation    Store training index files")
    print("        -JPEGImages                Store image files")
    print("        -SegmentationClass         Store label files")
    print('-' * 70)

def read_voc_images(voc_dir, is_train=True):
    """
    读取所有 VOC 特征和标签图像
    :param voc_dir: VOC 数据集目录
    :param is_train: 是否为训练集
    :return: 特征图像和标签图像的列表
    """
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, name in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{name}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{name}.png'), mode))
    return features, labels

def voc_color2label():
    """
    遍历 VOC 颜色映射，并将 RGB 值转换为类别索引,构建从 RGB 到 VOC 类别索引的映射
    :return: RGB 到类别索引的映射
    """
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)

    for i, colormap in enumerate(Z.VOC_COLOR):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

    return colormap2label

def voc_label2indices(colormap, colormap2label):
    """
    将 VOC 标签中的任意 RGB 值映射到类别索引
    :param colormap: 输入的颜色映射
    :param colormap2label: RGB 到类别索引的映射表
    :return: 类别索引
    """
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

def voc_randCrop(feature, label, height, width):
    """
    voc:随机裁剪特征图像和标签图像
    :param feature: 特征图像
    :param label: 标签图像
    :param height: 裁剪的目标高度
    :param width: 裁剪的目标宽度
    :return: 裁剪后的特征图像和标签图像
    """
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = TF.crop(feature, *rect)
    label = TF.crop(label, *rect)
    return feature, label

class VOCDataset(Dataset):
    """加载VOC数据集的自定义数据集"""
    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.color2label = voc_color2label()
        print('read ' + str(len(self.features)) + ' examples')

    def __len__(self):
        return len(self.features)

    def normalize_image(self, img):
        return self.transform(img / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
                img.shape[1] >= self.crop_size[0] and
                img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_randCrop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label2indices(label, self.color2label))


def load2voc(voc_dir, batch_size, crop_size, num_workers=4):
    """
    加载 VOC 语义分割数据集
    :param voc_dir: VOC 数据集目录
    :param batch_size: 批量大小
    :param crop_size: 裁剪尺寸
    :param num_workers: 工作线程数,一般是2到8,需要看自己的电脑配置
    :return: 训练集迭代器和测试集迭代器
    """
    train_iter = DataLoader(
        VOCDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = DataLoader(
        VOCDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter
