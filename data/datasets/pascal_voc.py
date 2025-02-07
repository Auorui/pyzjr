import os
from PIL import Image
import random
import numpy as np
from tqdm import tqdm
from pyzjr.utils.converter import to_2tuple
from pyzjr.data.datasets.baseset import BaseDataset
from pyzjr.nn.torchutils.OneHot import get_one_hot

VOC_COLOR = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
             [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
             [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
             [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
             [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
             [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def read_voc_images(voc_dir: str, is_train: bool):
    """
    读取 voc 数据集的图像并加载到列表
    :param voc_dir: str, VOCdevkit\VOC2007
    :param is_train: bool, 如果是 True 读取 train.txt, 否则读取 val.txt
    :return:
    """
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, name in enumerate(images):
        feature_path = os.path.join(voc_dir, 'JPEGImages', f'{name}.jpg')
        label_path = os.path.join(voc_dir, 'SegmentationClass', f'{name}.png')
        features.append(Image.open(feature_path).convert('RGB'))
        labels.append(Image.open(label_path).convert('RGB'))

    return features, labels

def voc_color2label():
    """
    遍历 VOC 颜色映射，并将 RGB 值转换为类别索引,构建从 RGB 到 VOC 类别索引的映射
    :return: RGB 到类别索引的映射
    """
    colormap2label = np.zeros(256 ** 3, dtype=np.int64)

    for i, colormap in enumerate(VOC_COLOR):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

    return colormap2label

def voc_label2indices(colormap, colormap2label):
    """
    将 VOC 标签中的任意 RGB 值映射到类别索引
    :param colormap: 输入的颜色映射
    :param colormap2label: RGB 到类别索引的映射表
    :return: 类别索引
    """
    colormap = np.array(colormap).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

class VOCSegmentation(BaseDataset):
    def __init__(
            self,
            root,
            num_classes=21,
            year="2007",
            is_train=True,
            transforms=None,
            input_shape=(480, 480),
    ):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2008", "2009", "2010", "2011", "2012"], \
            "year can only choose 2007 to 2012"
        assert num_classes <= 21, "Can only be used for datasets within 21 classification categories"
        self.year = year
        self.transforms = transforms
        self.input_shape = input_shape
        self.num_classes = num_classes
        voc_root = os.path.join(root, f"VOC{year}")
        assert os.path.exists(voc_root), "path '{}' does not exist.".format(voc_root)
        self.features, self.labels = read_voc_images(voc_root, is_train)

        self.assert_list_length(self.features, self.labels)
        self.color2label = voc_color2label()

    def __len__(self):
        return len(self.features)

    def assert_list_length(self, image, label):
        assert (len(image) == len(label)), r"The number of loaded images and labels does not match"

    def __getitem__(self, idx):
        image, label = self.features[idx], self.labels[idx]
        image, label = self.resizepadcenter(image, label, self.input_shape)
        if self.transforms:
            image, label = self.transforms(image, label)
        image = np.transpose(self.preprocess_input(image), (2, 0, 1))
        label = voc_label2indices(label, self.color2label)
        one_hot_label = np.transpose(get_one_hot(label, num_classes=self.num_classes), (2, 0, 1))
        return image, label, one_hot_label

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


if __name__ == '__main__':
    from pyzjr.augmentation.transforms import *
    data_path = r"D:\PythonProject\pytorch_segmentation_2D\MultiSegment\dataset\VOCdevkit"

    transforms = Compose2([
        RandomHorizontal_Flip(flip_prob=0.5),
        ToHSV(hue_range=(-0.1, 0.1), saturation_range=(0.6, 1.4), value_range=(0.7, 1.3)),
        ToTensor(),
        RandomContrast(),
        RandomBrightness(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = VOCSegmentation(data_path, transforms=transforms)
    for im, label in dataset:
        print(im.shape, label.shape)
        print(np.unique(label))