import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from pyzjr.data.datasets.custom_dataset import BaseDataset
from pyzjr.aug import resize_pad


def colormap2label(image, colormap, dtype="int64"):
    colormap_label = np.zeros(256 ** 3, dtype=np.int64)
    image = np.array(image, dtype=dtype)
    for i, color_map in enumerate(colormap):
        colormap_label[(color_map[0] * 256 + color_map[1]) * 256 + color_map[2]] = i
    idx = ((image[:, :, 0] * 256 + image[:, :, 1]) * 256
           + image[:, :, 2])
    return colormap_label[idx]

def generate_label_map(prediction, colormap):
    """
    Convert prediction to a GT map using the provided colors.
    Args:
        prediction (numpy.ndarray): The prediction array of shape (height, width). # pr = pr.argmax(axis=-1)  Get the class with the highest score for each pixel
        colormap (numpy.ndarray): The color palette.
    Returns:
        label_map (numpy.ndarray): A colored GT map.
    """
    h, w = prediction.shape[:2]
    image = np.zeros((h, w, 3), dtype='int32')
    for i in range(len(colormap)):
        image[prediction == i] = colormap[i]
    return image.astype(np.uint8)

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

def color_jitter(
        image: np.ndarray,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
) -> np.ndarray:
    """输入输出均为 float32 [0,1]  (RGB)"""
    if image.dtype == np.uint8 or np.max(image) > 1.0:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    # Brightness
    if brightness > 0:
        delta = np.random.uniform(-brightness, brightness)
        image = np.clip(image + delta, 0, 1)

    # Contrast
    if contrast > 0:
        alpha = 1.0 + np.random.uniform(-contrast, contrast)
        image = np.clip(alpha * image, 0, 1)

    # Saturation & Hue（float32 HSV: H∈[0,360), S,V∈[0,1]）
    if saturation > 0 or hue > 0:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # float32
        h, s, v = cv2.split(hsv)

        if saturation > 0:
            alpha = 1.0 + np.random.uniform(-saturation, saturation)
            s = np.clip(alpha * s, 0, 1)

        if hue > 0:
            # hue 参数按"弧度"语义 → 转成度数，OpenCV float32 H 是 [0,360)
            delta = np.random.uniform(-hue, hue) * 180.0 / np.pi
            h = (h + delta) % 360.0

        image = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)

    return np.clip(image, 0, 1).astype(np.float32)



class PascalVOCDataset(BaseDataset):
    def __init__(
            self,
            root="./VOCdevkit/VOC2012",
            num_classes=21,
            target_shape=256,
            is_train=True,
            color_map=VOC_COLOR,
            transforms=None,
            return_filename=False,
    ):
        super().__init__()
        txt_name = "train.txt" if is_train else "val.txt"
        self.seg_txt_path = os.path.join(root, r"ImageSets/Segmentation", txt_name)
        self.jpeg_images_path = os.path.join(root, r"JPEGImages")
        self.segmentation_class_path = os.path.join(root, r"SegmentationClass")
        self.target_shape = self.to_2tuple(target_shape)
        self.color_map = color_map
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
            feature = feature.astype(np.float32) / 255.0
        else:
            feature = self.read_image(feature_path, to_rgb=True, normalize=True)
            label = self.read_image(label_path, to_rgb=True, normalize=False)
            feature, label = self.segment_augument(feature, label)
        if self.color_map:
            label = colormap2label(label, self.color_map)
        feature = self.hwc2chw(feature)
        if self.return_filename:
            return feature, label, os.path.basename(feature_path)
        else:
            return feature, label

    def segment_augument(self, image, label):
        image, _ = resize_pad(image, self.target_shape, is_label=False)
        label, _ = resize_pad(label, self.target_shape, is_label=True)
        if self.is_train:
            image = color_jitter(image)
            image, label = self.augment([image, label], target_shape=None, is_labels=[False, True])
        return image, label

if __name__ == "__main__":
    import cv2
    root = r'E:\PythonProject\Pytorch_Segmentation_Auorui\data\VOCdevkit\VOC2012'
    train_dataset = PascalVOCDataset(root, is_train=True, color_map=VOC_COLOR)
    val_dataset = PascalVOCDataset(root, is_train=False, color_map=VOC_COLOR)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=2, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=2, pin_memory=True)
    print(len(train_dataset), len(val_dataset))

    # for feature, encoded_label in train_dataset:
    #     print(feature.shape, encoded_label.shape)
    #     print(np.unique(encoded_label))

    def display_dataset_samples(dataset, num_samples=3, random_order=True):
        """
        显示数据集中的图像、标签和融合图像
        """
        rows = num_samples
        cols = 3
        import random
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("TkAgg")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig, axs = plt.subplots(rows, cols, figsize=(12, rows*3.5),
                                gridspec_kw={'wspace':0.05, 'hspace':0.15})

        if random_order:
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        else:
            indices = range(min(num_samples, len(dataset)))

        for i, idx in enumerate(indices):
            feature, label = dataset[idx]

            # 处理图像：从 (3, H, W) 变成 (H, W, 3)
            image = np.transpose(feature, (1, 2, 0))
            # 如果归一化了，还原到 0-255
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

            # 处理标签
            if len(label.shape) == 3:
                label = label.squeeze()

            # 如果是类别索引图，直接涂色
            label_rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
            for class_idx, color in enumerate(VOC_COLOR):
                label_rgb[label == class_idx] = color

            # 确保尺寸严格一致
            if image.shape != label_rgb.shape:
                label_rgb = cv2.resize(label_rgb, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 融合图像 (注意：image 目前是 RGB，label_rgb 也是 RGB，在 RGB 通道下直接融合没问题)
            blended = cv2.addWeighted(image, 0.5, label_rgb, 0.5, 0)

            # 显示图像 (注意：matplotlib 默认接受 RGB 格式)
            axs[i, 0].imshow(image)
            axs[i, 0].set_title("原始图像", fontsize=10, pad=5)
            axs[i, 0].axis('off')

            axs[i, 1].imshow(label_rgb)
            axs[i, 1].set_title("语义标签", fontsize=10, pad=5)
            axs[i, 1].axis('off')

            axs[i, 2].imshow(blended)
            axs[i, 2].set_title("融合图像(0.5)", fontsize=10, pad=5)
            axs[i, 2].axis('off')

        plt.tight_layout(pad=1.5)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
        plt.suptitle("Pascal VOC 数据集样本展示", y=0.98, fontsize=12)
        plt.show()

    display_dataset_samples(train_dataset, num_samples=3)