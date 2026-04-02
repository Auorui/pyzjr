"""
The structure should be:
    data_dir/
        train/
            class1/
                img1.jpg
                img2.png
            class2/
                img3.jpg
                img4.png
        val/
        test/
"""
import os
import cv2
import torch
import numpy as np
from pyzjr.data.datasets.custom_dataset import BaseDataset

class ClassificationDataset(BaseDataset):
    def __init__(self, root_dir, target_shape, is_train=True, transforms=None, return_filename=False,
                 extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        super().__init__()
        self.root_dir = root_dir
        self.target_shape = self.to_2tuple(target_shape)
        self.is_train = is_train
        self.extensions = extensions
        self.transforms = transforms
        self.return_filename = return_filename
        self.path_and_label_list = self.scan_dataset(self.root_dir)

    def _is_valid_file(self, filename) -> bool:
        return filename.lower().endswith(self.extensions)

    def scan_dataset(self, root_dir):
        categories = self.natsorted(os.listdir(root_dir))
        self.num_classes = len(categories)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(categories)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        samples = []
        for cls_name in categories:
            cls_folder = os.path.join(self.root_dir, cls_name)
            for root, _, files in os.walk(cls_folder):
                for fname in files:
                    if self._is_valid_file(fname):
                        path = os.path.join(root, fname)
                        label = self.class_to_idx[cls_name]
                        samples.append((path, label))

        if not samples:
            raise RuntimeError(f"No valid images found in {self.root_dir}.")
        return samples

    def __len__(self):
        return len(self.path_and_label_list)

    def __getitem__(self, index):
        self.disable_cv2_multithreading()
        image_path, label = self.path_and_label_list[index]
        filename = os.path.basename(image_path)
        label = torch.tensor(label).long()
        image = self.read_image(image_path, to_rgb=True, normalize=False)
        if self.transforms:
            image = self.transforms(image)
        else:
            image = self.class_augument(image)
        if self.return_filename:
            return image, label, filename
        else:
            return image, label

    def class_augument(self, image):
        image = self.resizepad(image, target_shape=self.target_shape)
        if self.is_train:
            image = self.augment([image])[0]
            image = self.color_jitter(image)
        image = self.preprocess_input(image)
        image = self.hwc2chw(image)
        image = torch.from_numpy(image).float()
        return image

if __name__=="__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader
    data_path = r'F:\dataset\image_classification\base'
    class_train_dataset = ClassificationDataset(data_path, 256, is_train=True)
    train_dataset = ClassificationDataset(root_dir=os.path.join(data_path, 'train'), target_shape=[256, 256], is_train=True)
    val_dataset = ClassificationDataset(root_dir=os.path.join(data_path, 'val'), target_shape=[256, 256], is_train=False)
    print("训练集数量：", len(train_dataset), "训练集数量", len(val_dataset))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=4, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=2, pin_memory=True)

    # for batch in train_dataset:
    #     image, label = batch
    #     print(image.shape, label)

    transform = transforms.Compose([
        transforms.ToPILImage(),  # convert numpy to PIL image
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(size=(256, 256)),
        transforms.PILToTensor()
    ])
    train_dataset_transform = ClassificationDataset(root_dir=os.path.join(data_path, 'train'), target_shape=[256, 256], is_train=True,
                                          transforms=transform)

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

    show_image_from_dataloader(train_dataset_transform)