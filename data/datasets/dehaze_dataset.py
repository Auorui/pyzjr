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
        self.image_name_list = self.SearchFileName(self.gt, ('.png', '.jpg'))
        self.disable_cv2_multithreading()

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, item):
        img_name = self.image_name_list[item]
        # normalize to [-1, 1]
        source_img = self.read_image(os.path.join(self.hazy, img_name), to_rgb=True, normalize=True) * 2 - 1
        target_img = self.read_image(os.path.join(self.gt, img_name), to_rgb=True, normalize=True) * 2 - 1
        if self.mode:
            [source_img, target_img] = self.augment([source_img, target_img], target_shape=self.target_shape)
        else:
            [source_img, target_img] = self.align([source_img, target_img], self.target_shape)
        return self.hwc2chw(source_img), self.hwc2chw(target_img)