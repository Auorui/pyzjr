import os
import numpy as np
from torch.utils.data import DataLoader
from pyzjr.data.datasets.baseset import BaseDataset
from pyzjr.data.utils.tuplefun import to_2tuple
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

class PascalVOCSegDataset(BaseDataset):
    def __init__(
            self,
            root="./VOCdevkit/VOC2012",
            num_classes=21,
            target_shape=256,
            is_train=True,
            class_mapping=None,
            transforms=None,
    ):
        super(PascalVOCSegDataset, self).__init__()
        txt_name = "train.txt" if is_train else "val.txt"
        self.seg_txt_path = os.path.join(root, r"ImageSets/Segmentation", txt_name)
        self.jpeg_images_path = os.path.join(root, r"JPEGImages")
        self.segmentation_class_path = os.path.join(root, r"SegmentationClass")
        self.target_shape = to_2tuple(target_shape)
        self.class_mapping = class_mapping
        self.image_names = self.read_txt(self.seg_txt_path)
        self.num_classes = num_classes
        self.transforms = transforms

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        self.disable_cv2_multithreading()
        feature_path = os.path.join(self.jpeg_images_path, f'{self.image_names[item]}.jpg')
        label_path = os.path.join(self.segmentation_class_path, f'{self.image_names[item]}.png')
        feature = self.read_image(feature_path, target_shape=self.target_shape,
                                  to_rgb=True, normalize=True)
        label = self.read_image(label_path, target_shape=self.target_shape,
                                to_rgb=True, normalize=False, pad_color=(0, 0, 0))
        if self.transforms:
            feature, label = self.transforms(feature, label)
        else:
            feature, label = self.auguments([feature, label])
        if self.class_mapping:
            label = self.colormap2label(label, self.class_mapping)
        feature = self.hwc2chw(feature)
        encoded_label = self.hwc2chw(get_one_hot(label, num_classes=self.num_classes))

        return feature, encoded_label

if __name__ == "__main__":
    root = r'E:\PythonProject\Pytorch_Segmentation_Auorui\data\VOCdevkit\VOC2012'
    train_dataset = PascalVOCSegDataset(root, is_train=True, class_mapping=VOC_COLOR)
    # display_dataset_samples(voc_dataset, num_samples=3)
    val_dataset = PascalVOCSegDataset(root, is_train=False, class_mapping=VOC_COLOR)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=2, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=2, pin_memory=True)
    print(len(train_dataset), len(val_dataset))
    # for feature, encoded_label in train_loader:
    #     print(feature.shape, encoded_label.shape)
    #     print(np.unique(encoded_label))



    def display_dataset_samples(dataset, num_samples=3):
        """
        显示数据集中的图像和标签。

        Args:
            dataset (PascalVOCSegDataset): 数据集对象。
            num_samples (int): 要显示的样本数量。
        """
        rows = num_samples
        cols = 2
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("TkAgg")
        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, rows * 4))

        for i in range(num_samples):
            feature, encoded_label = dataset[i]

            image = np.transpose(feature, (1, 2, 0))  # CHW -> HWC
            label = np.argmax(encoded_label, axis=0)  # One-Hot -> 类别索引

            ax[i, 0].imshow(image)
            ax[i, 0].set_title("Image")
            ax[i, 0].set_axis_off()

            ax[i, 1].imshow(label, cmap="jet", interpolation="nearest")
            ax[i, 1].set_title("Ground Truth Mask")
            ax[i, 1].set_axis_off()

        plt.tight_layout()
        plt.show()