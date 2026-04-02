import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from pyzjr.data.datasets.custom_dataset import BaseDataset
from pyzjr.nn.strategy import colormap2label
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

class PascalVOCDataset(BaseDataset):
    def __init__(
            self,
            root="./VOCdevkit/VOC2012",
            num_classes=21,
            target_shape=256,
            is_train=True,
            class_mapping=VOC_COLOR,
            transforms=None,
            return_filename=False,
    ):
        super().__init__()
        txt_name = "train.txt" if is_train else "val.txt"
        self.seg_txt_path = os.path.join(root, r"ImageSets/Segmentation", txt_name)
        self.jpeg_images_path = os.path.join(root, r"JPEGImages")
        self.segmentation_class_path = os.path.join(root, r"SegmentationClass")
        self.target_shape = self.to_2tuple(target_shape)
        self.class_mapping = class_mapping
        self.is_train = is_train
        self.image_names = self.read_txt(self.seg_txt_path)
        self.num_classes = num_classes
        self.transforms = transforms
        self.return_filename = return_filename

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        self.disable_cv2_multithreading()
        feature_path = os.path.join(self.jpeg_images_path, f'{self.image_names[item]}.jpg')
        label_path = os.path.join(self.segmentation_class_path, f'{self.image_names[item]}.png')
        if self.transforms:
            feature = Image.open(feature_path).convert('RGB')
            label = Image.open(label_path).convert('RGB')
            feature, label = self.transforms(feature, label)
            feature = self.preprocess_input(feature)
        else:
            feature = self.read_image(feature_path, to_rgb=True, normalize=True)
            label = self.read_image(label_path, to_rgb=True, normalize=False)
            feature, label = self.segment_augument(feature, label)
        if self.class_mapping:
            label = colormap2label(label, self.class_mapping)
        feature = self.hwc2chw(feature)
        encoded_label = self.hwc2chw(get_one_hot(label, num_classes=self.num_classes))
        if self.return_filename:
            return feature, encoded_label, os.path.basename(feature_path)
        else:
            return feature, encoded_label

    def segment_augument(self, image, label):
        image, label = self.resizepad(image, self.target_shape, label=label)
        if self.is_train:
            image, label = self.augment([image, label])
        return image, label

if __name__ == "__main__":
    import cv2
    root = r'E:\PythonProject\Pytorch_Segmentation_Auorui\data\VOCdevkit\VOC2012'
    train_dataset = PascalVOCDataset(root, is_train=True, class_mapping=VOC_COLOR)
    val_dataset = PascalVOCDataset(root, is_train=False, class_mapping=VOC_COLOR)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=2, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=2, pin_memory=True)
    print(len(train_dataset), len(val_dataset))

    from pyzjr.augmentation.transforms import tvisionColorJitter, tvisionPerspectiveTransform, tvisionResizePad, Compose, NdArraytoPIL, PILtoNdArray

    transform = Compose([
        # 推荐仅做图像处理，不要使用ToTensor
        # NdArraytoPIL(),
        # tvisionRandomResize(min_size=256, max_size=512),
        # tvisionRandomHorizontalFlip(prob=0.5),
        # tvisionRandomCrop(size=256),
        # tvisionCenterCrop(size=200),
        tvisionColorJitter(),
        tvisionPerspectiveTransform(0.6),
        tvisionResizePad(size=256),
        PILtoNdArray(),
    ])
    train_dataset_transformer = PascalVOCDataset(root, is_train=True, class_mapping=VOC_COLOR, transforms=transform)

    # for feature, encoded_label in train_dataset:
    #     print(feature.shape, encoded_label.shape)
    #     print(np.unique(encoded_label))

    def display_dataset_samples(dataset, num_samples=3, random_order=True):
        """
        显示数据集中的图像、标签和融合图像，保持紧凑布局但有适当间隙

        Args:
            dataset (PascalVOCSegDataset): 数据集对象
            num_samples (int): 要显示的样本数量
        """
        rows = num_samples
        cols = 3  # 图像、标签和融合图像三列
        import random
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False
        fig, axs = plt.subplots(rows, cols, figsize=(12, rows*3.5),
                                gridspec_kw={'wspace':0.05, 'hspace':0.15})
        if random_order:
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        else:
            indices = range(min(num_samples, len(dataset)))
        for i, idx in enumerate(indices):
            feature, encoded_label = dataset[idx]
            image = np.transpose(feature, (1, 2, 0))
            if image.max() <= 1.0:  # 归一化图像
                image = (image * 255).astype(np.uint8)
            label = np.argmax(encoded_label, axis=0)
            label_rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
            for class_idx, color in enumerate(VOC_COLOR):
                label_rgb[label == class_idx] = color
            # 0.5
            blended = cv2.addWeighted(image, 0.5, label_rgb, 0.5, 0)
            axs[i, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axs[i, 0].set_title("原始图像", fontsize=10, pad=5)
            axs[i, 0].axis('off')

            axs[i, 1].imshow(label_rgb)
            axs[i, 1].set_title("语义标签", fontsize=10, pad=5)
            axs[i, 1].axis('off')

            axs[i, 2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            axs[i, 2].set_title("融合图像(0.5)", fontsize=10, pad=5)
            axs[i, 2].axis('off')

        plt.tight_layout(pad=1.5)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
        plt.suptitle("Pascal VOC 数据集样本展示", y=0.98, fontsize=12)
        plt.show()


    display_dataset_samples(train_dataset, num_samples=3)