"""
数据集标准如下:
    - base                          - 数据集名称
        - train                         - 训练集
            - crack                         - 类别名 1
            - rust                          - 类别名 2
            - spalling                      - 类别名 3
            - stoma                         - 类别名 4
        - val                           - 验证集
        - test                          - 测试集（ 不一定需要 ）
"""
import os
import cv2
import torch
import random
import numpy as np
from pathlib import Path
from pyzjr.data.utils import natsorted
from pyzjr.data.datasets.baseset import BaseDataset

class ClassificationDataset(BaseDataset):
    def __init__(self, root_dir, target_shape, is_train=True, transform=None):
        super(ClassificationDataset, self).__init__()
        data_folder = os.path.join(root_dir, 'train') if is_train else os.path.join(root_dir, 'val')
        self.categories = natsorted(os.listdir(data_folder))
        self.num_classes = len(self.categories)
        if isinstance(target_shape, (list, tuple)) and len(target_shape) == 2:
            self.target_shape = target_shape
        elif isinstance(target_shape, int):
            self.target_shape = (target_shape, target_shape)
        self.transform = transform
        self.data_and_label_list = []
        self.is_train = is_train
        for label, category in enumerate(self.categories):
            category_folder = os.path.join(data_folder, category)
            self.data_and_label_list.extend(
                [(str(f), label) for f in Path(category_folder).iterdir() if f.is_file() and not f.name.startswith(".")])

    def __len__(self):
        return len(self.data_and_label_list)

    def __getitem__(self, index):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        image_path, label = self.data_and_label_list[index]
        label = torch.tensor(label).long()
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.transform:
            image = self.transform(image)
        else:
            image = self.class_augument(image, self.target_shape, is_train=self.is_train)
            image = np.transpose(image, [2, 0, 1])
            image = torch.from_numpy(image).float()
        return image, label

    def class_augument(self, image, target_shape, is_train, prob=.5):
        h, w = target_shape
        ih, iw = image.shape[:2]
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        new_image = np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)
        top = (h - nh) // 2
        left = (w - nw) // 2
        new_image[top:top + nh, left:left + nw] = resized_image
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        if is_train:
            if random.random() > (1 - prob):
                new_image = np.flip(new_image, axis=1)
            r = random.randint(0, 3)
            new_image = np.rot90(new_image, r, (0, 1))
            # 转换到HSV颜色空间
            hsv_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV).astype(np.float32)
            h_channel, s_channel, v_channel = cv2.split(hsv_image)
            # 随机调整色调（Hue）
            delta_h = random.randint(-30, 30)
            h_channel = (h_channel + delta_h) % 180
            # 随机调整饱和度（Saturation）和亮度（Value）
            s_scale = np.random.uniform(0.5, 1.5)
            v_scale = random.uniform(0.5, 1.5)
            s_channel = np.clip(s_channel * s_scale, 0, 255).astype(np.uint8)
            v_channel = np.clip(v_channel * v_scale, 0, 255).astype(np.uint8)
            # 合并通道并转换回RGB颜色空间
            hsv_image[..., 0] = h_channel.astype(np.uint8)
            hsv_image[..., 1] = s_channel
            hsv_image[..., 2] = v_channel
            new_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2RGB)
        # 将图像归一化到[0, 1]范围
        new_image = new_image / 255.0
        return new_image


if __name__=="__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader
    data_path = r'E:\PythonProject\pytorch_classification_Auorui\data\cat_dog'
    train_dataset = ClassificationDataset(root_dir=data_path, target_shape=[256, 256], is_train=True)
    val_dataset = ClassificationDataset(root_dir=data_path, target_shape=[256, 256], is_train=False)
    print("训练集数量：", len(train_dataset), "训练集数量", len(val_dataset))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=4, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=2, pin_memory=True)
    # for batch in val_dataset:
    #     image, label = batch
    #     print(image.shape, label)
    # show_image_from_dataloader(train_dataset)


    transform = transforms.Compose([
        transforms.ToPILImage(),  # convert numpy to PIL image
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(size=(256, 256)),
        transforms.PILToTensor()
    ])
    train_dataset = ClassificationDataset(root_dir=data_path, target_shape=[256, 256], is_train=True,
                                          transform=transform)
    # show_image_from_dataloader(train_dataset)
    # for batch in train_dataset:
    #     image, label = batch
    #     print(image.shape, label)

    def show_image_from_dataloader(test_dataset):
        loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            pin_memory=False,
            drop_last=True
        )
        for i, (img, label) in enumerate(loader):
            img = img[0].numpy()
            img = np.transpose(img, (1, 2, 0))
            img = (img*255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(img, f"{label.item()}", (30, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, thickness=2, color=(255, 0, 255))
            cv2.imshow('show image from dataloader', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()