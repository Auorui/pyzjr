"""
去雾数据集专用格式如下:
    - rshazy
        - test
        - train
            - GT
                1.png
                2.png
            - hazy
                1.png
                2.png
                ...
        - test.txt
            1
            2
            ...
        - train.txt
"""
import os
import cv2
import random
import numpy as np
from pyzjr.data.utils.tuplefun import to_2tuple
from pyzjr.data.datasets.baseset import BaseDataset

class DeHazeDataset(BaseDataset):
    def __init__(
            self,
            root_dir,  # Synscapes
            target_shape,
            is_train=True,
            edge_decay=0,
            only_h_flip=False
    ):
        super(DeHazeDataset, self).__init__()
        self.mode = is_train
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip
        self.target_shape = to_2tuple(target_shape)
        self.data_dir = os.path.join(root_dir, 'train' if is_train else 'test')
        txt_path = self.data_dir + '.txt'
        self.txt_content = self.read_txt(txt_path)

    def __len__(self):
        return len(self.txt_content)

    def __getitem__(self, item):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img_name = f"{self.txt_content[item]}.png"
        # 归一化到 [-1, 1]
        source_img = self.read_image(os.path.join(self.data_dir, 'hazy', img_name)) * 2 - 1
        target_img = self.read_image(os.path.join(self.data_dir, 'GT', img_name)) * 2 - 1

        if self.mode:
            [source_img, target_img] = self.augment([source_img, target_img], self.target_shape, self.edge_decay, self.only_h_flip)
        else:
            [source_img, target_img] = self.align([source_img, target_img], self.target_shape)

        return self.hwc2chw(source_img), self.hwc2chw(target_img)


    def augment(self, imglists, size, edge_decay=0., only_h_flip=False):
        H, W, _ = imglists[0].shape
        Hc, Wc = size

        # simple re-weight for the edge
        if random.random() < Hc / H * edge_decay:
            Hs = 0 if random.randint(0, 1) == 0 else H - Hc
        else:
            Hs = random.randint(0, H - Hc)

        if random.random() < Wc / W * edge_decay:
            Ws = 0 if random.randint(0, 1) == 0 else W - Wc
        else:
            Ws = random.randint(0, W - Wc)

        for i in range(len(imglists)):
            imglists[i] = imglists[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

        # horizontal flip
        if random.randint(0, 1) == 1:
            for i in range(len(imglists)):
                imglists[i] = np.flip(imglists[i], axis=1).copy()

        if not only_h_flip:
            # bad data augmentations for outdoor
            rot_deg = random.randint(0, 3)
            for i in range(len(imglists)):
                imglists[i] = np.rot90(imglists[i], rot_deg, (0, 1)).copy()

        return imglists


    def align(self, imglists, size):
        H, W, _ = imglists[0].shape
        Hc, Wc = size

        Hs = (H - Hc) // 2
        Ws = (W - Wc) // 2
        for i in range(len(imglists)):
            imglists[i] = imglists[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

        return imglists


if __name__=="__main__":
    from torch.utils.data import DataLoader
    data_dir = r"rshazy"
    train_dataset = DeHazeDataset(data_dir, target_shape=256, is_train=True)
    val_dataset = DeHazeDataset(data_dir, target_shape=256,
                                is_train=False)
    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=2,
                            pin_memory=True)
    for i , (gt, hazy) in enumerate(train_loader):
        print(i, gt.shape, hazy.shape)
    for i , (gt, hazy) in enumerate(val_loader):
        print(i, gt.shape, hazy.shape)