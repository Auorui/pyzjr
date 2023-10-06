import os
import random
import numpy as np
import torchvision
import torch
import pyzjr.Z as Z
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from pyzjr.dlearn.strategy import cvtColor

def getPalette(color=Z.VOC_COLOR):
    pal = np.array(color, dtype='uint8').flatten()
    return pal

def voc_catalog(year = '2007', creat=False):
    assert year in ["2007","2012"], "year can only choose 2007 and 2012"
    print('VOC Catalog:')
    print('-' * 70)
    print("VOCdevkit")
    print(f"    VOC{year}")
    print("        -ImageSets/Segmentation    Store training index files")
    print("        -JPEGImages                Store image files")
    print("        -SegmentationClass         Store label files")
    print('-' * 70)
    if creat:
        os.makedirs(f'VOCdevkit/VOC{year}/ImageSets/Segmentation', exist_ok=True)
        os.makedirs(f'VOCdevkit/VOC{year}/JPEGImages', exist_ok=True)
        os.makedirs(f'VOCdevkit/VOC{year}/SegmentationClass', exist_ok=True)
        print("[pyzjr]:Successfully created!")


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
            voc_dir, 'SegmentationClass', f'{name}.png'), mode))
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
    """VOC数据集的自定义数据集加载器"""
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


def voc_annotation(trainval_percent=1, train_percent=0.9, VOCdevkit_path = 'VOCdevkit'):
    """
    当前将测试集当作验证集使用，不单独划分测试集
    :param trainval_percent: 想要增加测试集修改trainval_percent
    :param train_percent: 用于改变验证集的比例 9:1
    :param VOCdevkit_path: 指向VOC数据集所在的文件夹,默认指向根目录下的VOC数据集
    :return:
    """
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')

    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num = len(total_seg)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("traub suze", tr)
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in list:
        name = total_seg[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    print("检查数据集格式是否符合要求，这可能需要一段时间。")
    classes_nums = np.zeros([256], int)
    for i in tqdm(list):
        name = total_seg[i]
        png_file_name = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("未检测到标签图片%s，请查看具体路径下文件是否存在以及后缀是否为png。" % (png_file_name))

        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("标签图片%s的shape为%s，不属于灰度图或者八位彩图，请仔细检查数据集格式。" % (name, str(np.shape(png))))
            print("标签图片需要为灰度图或者八位彩图，标签的每个像素点的值就是这个像素点所属的种类。")

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |" % ("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |" % (str(i), str(classes_nums[i])))
            print('-' * 37)

    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("检测到标签中像素点的值仅包含0与255，数据格式有误。")
        print("二分类问题需要将标签修改为背景的像素点值为0，目标的像素点值为1。")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("检测到标签中仅仅包含背景像素点，数据格式有误，请仔细检查数据集格式。")

    print("JPEGImages中的图片应当为.jpg文件、SegmentationClass中的图片应当为.png文件。")


class VOCSegmentation(Dataset):
    def __init__(self,voc_root='VOCdevkit',year='2007',transforms=None, txt_name = "train.txt"):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007","2012"], "year can only choose 2007 and 2012"
        root = os.path.join(voc_root, f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)

        imgs_dir = os.path.join(root, 'JPEGImages')
        masks_dir = os.path.join(root, 'SegmentationClass')
        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.images = [os.path.join(imgs_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(masks_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index])
        img = cvtColor(img)
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
