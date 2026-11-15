"""
seg dataset
    - train
        - images
        - labels
    - val
    - test
"""
import os
import cv2
import torch
import numpy as np

from pyzjr.aug.geometric import resize_pad
from pyzjr.data.helper import label2index
from pyzjr.data.datasets.custom_dataset import BaseDataset


class SegDataset(BaseDataset):
    """Unified segmentation dataset for binary and multi-class labels.

    Label handling:
        - (H,W) int index map       -> returned as int64 directly
        - (H,W) grayscale map       -> thresholded to binary
        - (H,W,3) color label map   -> mapped to indices via color_map
          (binary case with >2 unique colors is thresholded first)
    """
    def __init__(
            self,
            root_dir,
            target_shape,
            num_classes,
            mode="train",
            dir_image="images",
            dir_label="labels",
            color_map=None,
            label_threshold=127,
            ignore_index=256,
            strict=True,
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if mode not in ('train', 'val', 'test'):
            raise ValueError(f"mode must be 'train'/'val'/'test', got {mode!r}")

        self.mode = mode
        self.num_classes = int(num_classes)
        self.target_shape = self.to_2tuple(target_shape)
        self.ignore_index = int(ignore_index)
        self.strict = strict
        self.label_threshold = int(label_threshold)

        data_dir = os.path.join(root_dir, mode)
        self.images_dir = os.path.join(data_dir, dir_image)
        self.labels_dir = os.path.join(data_dir, dir_label)

        suffixes = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        self.image_name_list = self.get_file_path(self.images_dir, suffixes)

        self.color_map = None
        if color_map is not None:
            # normalize color_map: int keys, uint8 (3,) values
            self.color_map = {}
            for index, (key, color) in enumerate(color_map.items()):
                class_id = int(key) if str(key).isdigit() else index
                if not 0 <= class_id < self.num_classes:
                    raise ValueError(f"Invalid class id: {class_id}")
                color = np.asarray(color, dtype=np.uint8)
                if color.shape != (3,):
                    raise ValueError(f"Color for class {class_id} must be RGB triplet")
                self.color_map[class_id] = color
        elif self.num_classes == 2:
            self.color_map = {0: np.array([0, 0, 0], np.uint8), 1: np.array([255, 255, 255], np.uint8)}

        self.disable_cv2_multithreading()

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, index):
        image_name = self.image_name_list[index]

        image = self.read_image(
            os.path.join(self.images_dir, image_name),
            to_rgb=True, normalize=True,
        )
        label = self.read_image(
            os.path.join(self.labels_dir, image_name),
            to_rgb=True, normalize=False,
        )

        assert image.shape[:2] == label.shape[:2], \
            f"image {image.shape[:2]} and label {label.shape[:2]} size mismatch: {image_name}"

        pad_meta = None
        if self.mode == "train":
            image, label = self.augment(
                [image, label], self.target_shape, is_labels=[False, True]
            )
        elif self.mode == 'val':
            image, label = self.align(
                [image, label], self.target_shape, is_labels=[False, True], mode='resize_pad'
            )
        else:
            image, pad_meta = resize_pad(image, self.target_shape, is_label=False)
            label, _ = resize_pad(label, self.target_shape, is_label=True)

        image_tensor = torch.from_numpy(self.hwc2chw(image)).float()
        label_tensor = torch.from_numpy(self.process_label(label)).long()

        if self.mode == 'test':
            return image_tensor, label_tensor, pad_meta, image_name

        return image_tensor, label_tensor


    def process_label(self, label):
        if self.color_map is None:
            label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
        return label2index(
            label,
            num_classes=self.num_classes,
            color_map=self.color_map,
            ignore_index=self.ignore_index,
            label_threshold=self.label_threshold,
            strict=self.strict
        )

if __name__ == "__main__":
    import random
    from torch.utils.data import DataLoader

    train_dataset = SegDataset(
        root_dir=r'E:\PythonProject\SegLab\data\Concrete400\train',
        target_shape=256,
        num_classes=2,
        mode='val',
        color_map={0: [0, 0, 0], 1: [255, 255, 255]},
        label_threshold=127,
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    for i, (image, label) in enumerate(train_loader):
        print(image.shape, label.shape)
        print(f"Unique values in batch {i}: {torch.unique(label)}")

    random_idx = random.randint(0, len(train_dataset) - 1)
    print(f"\n正在可视化第 {random_idx} 张图片...")
    image_tensor, label_tensor = train_dataset[random_idx]

    image = image_tensor.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    label = label_tensor.numpy().astype(np.uint8)
    label_rgb = np.zeros_like(image)
    label_rgb[label == 1] = [255, 0, 0]
    blended = cv2.addWeighted(image, 0.6, label_rgb, 0.4, 0)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    label_bgr = cv2.cvtColor(label_rgb, cv2.COLOR_RGB2BGR)
    combined = np.hstack([image_bgr, label_bgr, blended_bgr])

    cv2.imshow("Original | Mask (Red=1) | Overlay", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()