"""
2 num_classes
# CLCD dataset
    - train
        - label                label image in training set
        - image1               image at t1 in training set
        - image2               image at t2 in training set
    - val
    - test
"""
import cv2
import os
import torch
import numpy as np

from pyzjr.aug.geometric import resize_pad
from pyzjr.data.helper import label2index
from pyzjr.data.datasets.custom_dataset import BaseDataset

class CdDataset(BaseDataset):
    def __init__(
            self,
            root_dir,
            target_shape,
            num_classes=2,
            mode='train',
            dir_n1='image1',
            dir_n2='image2',
            dir_label='label',
            color_map=None,
            label_threshold=127,
            ignore_index=256,
            strict=True,
            val_mode='resize_pad'  # center_crop
    ):
        super(CdDataset, self).__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if mode not in ('train', 'val', 'test'):
            raise ValueError(f"mode must be 'train'/'val'/'test', got {mode!r}")
        self.mode = mode
        self.num_classes = num_classes
        self.target_shape = self.to_2tuple(target_shape)
        data_dir = os.path.join(root_dir, self.mode)
        self.t1 = os.path.join(data_dir, dir_n1)
        self.t2 = os.path.join(data_dir, dir_n2)
        self.label = os.path.join(data_dir, dir_label)

        suffixes = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        self.image_name_list = self.get_file_path(self.t1, suffixes)

        self.ignore_index = int(ignore_index)
        self.label_threshold = int(label_threshold)
        self.strict = strict
        self.val_mode = val_mode

        self.color_map = None
        if color_map is not None:
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
        img_name = self.image_name_list[index]

        img1 = self.read_image(os.path.join(self.t1, img_name), to_rgb=True, normalize=True)
        img2 = self.read_image(os.path.join(self.t2, img_name), to_rgb=True, normalize=True)
        label = self.read_image(os.path.join(self.label, img_name), to_rgb=True, normalize=False)

        pad_meta = None
        if self.mode == 'train':
            [img1, img2, label] = self.augment(
                [img1, img2, label], self.target_shape, prob=.5, is_labels=[False, False, True])
        elif self.mode == 'val':
            img1, img2, label = self.align(
                [img1, img2, label], self.target_shape, is_labels=[False, False, True], mode=self.val_mode)
        else:
            img1, pad_meta = resize_pad(img1, self.target_shape, is_label=False)
            img2, _ = resize_pad(img2, self.target_shape, is_label=False)
            label, _ = resize_pad(label, self.target_shape, is_label=True)

        img1 = torch.from_numpy(self.hwc2chw(img1)).float()
        img2 = torch.from_numpy(self.hwc2chw(img2)).float()
        label = torch.from_numpy(self.process_label(label)).long()
        if self.mode == 'test':
            return img1, img2, label, pad_meta, img_name
        return img1, img2, label

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


def build_dataset(config, mode='train'):
    """Build the dataset declared in a loaded config."""
    data = config.data if hasattr(config, "data") else config
    root_dir = str(data.dataset_path)
    target_shape = data.target_shape
    num_classes = int(data.num_classes)
    image1_dir = data.dir_n1
    image2_dir = data.dir_n2
    label_dir = data.dir_label
    color_map = getattr(data, "color_map", None)
    label_threshold = getattr(data, "label_threshold", 127)
    ignore_index = getattr(data, "ignore_index", 256)
    val_mode = getattr(data, "val_mode", 'resize_pad')

    return CdDataset(
        root_dir,
        target_shape,
        num_classes,
        mode=mode,
        dir_n1=image1_dir,
        dir_n2=image2_dir,
        dir_label=label_dir,
        color_map=color_map,
        label_threshold=label_threshold,
        ignore_index=ignore_index,
        val_mode=val_mode
    )


if __name__ == "__main__":
    clcd_dir = r'E:\PythonProject\CDLab\data\CLCD_s'
    crop_dir = r'E:\PythonProject\CDLab\data\CropSCD_s'

    # train_set = CdDataset(clcd_dir, 512, mode='train')
    train_set = CdDataset(crop_dir, 512, 9, mode='train', dir_n1='im1', dir_n2='im2')

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=2, num_workers=2, shuffle=True)

    if train_set.num_classes == 2:
        palette = np.zeros((max(2, int(train_set.ignore_index) + 1), 3), np.uint8)
        palette[0] = [0, 0, 0]  # 未变化：黑
        palette[1] = [255, 255, 255]  # 变化：白
    else:
        palette = np.zeros((max(train_set.num_classes, int(train_set.ignore_index) + 1), 3), np.uint8)
        if train_set.color_map is not None:
            for cid, color in train_set.color_map.items():
                palette[cid] = np.asarray(color, np.uint8)
        else:
            for cid in range(train_set.num_classes):
                palette[cid] = [(cid * 97) % 256, (cid * 57) % 256, (cid * 23) % 256]

    for i, (image1, image2, label) in enumerate(train_loader):
        print(image1.shape, image2.shape, label.shape)
        print(f"Unique values in batch {i}: {torch.unique(label)}")

        img1 = image1[0].numpy()   # (3, H, W)
        img2 = image2[0].numpy()
        lbl = label[0].numpy()     # (H, W)

        # (C, H, W) -> (H, W, C)
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))

        img1_uint8 = (img1 * 255).astype(np.uint8)
        img2_uint8 = (img2 * 255).astype(np.uint8)

        img1_uint8 = img1_uint8[..., ::-1]
        img2_uint8 = img2_uint8[..., ::-1]

        lbl = lbl.astype(np.int64)
        lbl_rgb = palette[np.clip(lbl, 0, len(palette) - 1)]  # (H,W,3) RGB
        lbl_bgr = lbl_rgb[..., ::-1]

        stack_img = np.hstack([img1_uint8, img2_uint8, lbl_bgr])

        stack_img = cv2.resize(stack_img, (768, 256))
        cv2.imshow('T1 | T2 | Label (Press any key to next, ESC to exit)', stack_img)

        k = cv2.waitKey(0)
        if k == 27:
            break

    cv2.destroyAllWindows()