"""
The dehazing dataset specific format is as follows:
    - RESIDE-IN
        - train
            - GT
            - hazy
        - test
            - GT
            - hazy
Use ITS as the training set and SOTS indoor as the test set.
"""
import os
from pyzjr.data.datasets.custom_dataset import BaseDataset

class DehazeDataset(BaseDataset):
    def __init__(
            self,
            root_dir,
            target_shape,
            is_train=True,
    ):
        super(DehazeDataset, self).__init__()
        self.mode = is_train
        self.target_shape = self.to_2tuple(target_shape)
        data_dir = os.path.join(root_dir, 'train' if is_train else 'test')
        self.gt = os.path.join(data_dir, 'GT')
        self.hazy = os.path.join(data_dir, 'hazy')
        self.image_name_list = self.get_file_path(self.gt)
        self.disable_cv2_multithreading()

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, item):
        img_name = self.image_name_list[item]
        # normalize to [-1, 1]
        source_img = self.read_image(os.path.join(self.hazy, img_name), to_rgb=True, normalize=True) * 2 - 1
        target_img = self.read_image(os.path.join(self.gt, img_name), to_rgb=True, normalize=True) * 2 - 1
        if self.mode:
            [source_img, target_img] = self.augment([source_img, target_img], target_shape=self.target_shape, is_labels=[False, False])
        else:
            [source_img, target_img] = self.align([source_img, target_img], target_shape=self.target_shape, is_labels=[False, False])
        return self.hwc2chw(source_img), self.hwc2chw(target_img)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    data_path = r'E:\PythonProject\DehazeLab\data\RSHD\thick'
    train_dataset= DehazeDataset(
        root_dir=data_path,
        target_shape=(256, 256),
        is_train=True,
    )
    val_dataset=DehazeDataset(
        root_dir=data_path,
        target_shape=256,
        is_train=False,
    )

    train_loader=DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    val_loader=DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=False
    )

    import cv2
    import numpy as np

    for i, (hazy, gt) in enumerate(val_loader):
        gt_img = gt[0].numpy()
        hazy_img = hazy[0].numpy()
        gt_img = np.transpose(gt_img, axes=[1, 2, 0])
        hazy_img = np.transpose(hazy_img, axes=[1, 2, 0])
        gt_img = ((gt_img + 1)*127.5)[:,:,::-1].astype(np.uint8)
        hazy_img = ((hazy_img + 1)*127.5)[:,:,::-1].astype(np.uint8)
        stack_img = np.hstack([gt_img, hazy_img])
        cv2.imshow('GT & Hazy', stack_img)
        k=cv2.waitKey(0)
        if k == 27:   # ESC exit
            cv2.destroyAllWindows()
            break