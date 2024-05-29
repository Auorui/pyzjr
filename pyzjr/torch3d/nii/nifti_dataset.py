import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pyzjr.torch3d.nii.nii_utils import nii_load, center_scale_nii, center_crop_nii
from pyzjr.nn.torchutils.OneHot import one_hot_3d

class NiftiDataset3D(Dataset):
    """
    用于加载3D Nifti格式图像数据的PyTorch数据集类
    默认是采用了中心裁剪和中心缩放的方式来减少内存需求，所以，
    ‘ target_shape ’ = 裁剪大小 * 缩放比例， 一般请勿修改缩放比例

    Args:
        image_path (str): 包含图像文件的目录路径。
        label_path (str): 包含标签文件的目录路径。
        target_shape (tuple): 所需的图像和标签的目标形状。
        scale (float): 缩放因子，用于缩放图像和标签。默认为0.5.
        num_classes (int): 类别数量。
        normalize (bool): 是否对图像进行归一化。默认为True.
    """
    def __init__(self,
                 image_path,
                 label_path,
                 target_shape=(160, 160, 160),
                 scale=.6,
                 num_classes=None,
                 normalize=True
                 ):
        super().__init__()
        self.image_paths = self.read_path2list(image_path)
        self.label_paths = self.read_path2list(label_path)
        self.num_classes = num_classes
        self.normalize = normalize
        self.scale = scale
        self.target_crop_shape = target_shape

    def __getitem__(self, item):
        one_image, one_label = self.read_nii_im_and_label(
            self.image_paths[item], self.label_paths[item]
        )
        one_image = center_crop_nii(one_image, self.target_crop_shape)
        one_label = center_crop_nii(one_label, self.target_crop_shape)
        one_image = center_scale_nii(one_image, self.scale)
        one_label = center_scale_nii(one_label, self.scale)
        # z, y, x = one_image.shape
        if self.normalize:
            one_image = self.normalize_image(one_image)

        image_trans = one_image[np.newaxis, :, :, :]  # b c w w h
        self.label_trans = one_label[np.newaxis, :, :, :]  # b c d w h

        if isinstance(image_trans, torch.Tensor):
            image_trans = image_trans.numpy()
        if isinstance(self.label_trans, torch.Tensor):
            self.label_trans = self.label_trans.numpy()
        one_hot_label_trans = one_hot_3d(self.label_trans[0], num_classes=self.num_classes)
        # one_hot_label_trans  d  y  x  c ----->  c  d  y  x
        return image_trans, np.transpose(one_hot_label_trans, (3, 0, 1, 2))

    def read_nii_im_and_label(self, image_path, label_path):
        image, _, _ = nii_load(image_path)
        label, _, _ = nii_load(label_path)
        # w, h, d  ----->  d, h, w
        return np.transpose(image, (2, 1, 0)), np.transpose(label, (2, 1, 0))

    def __len__(self):
        return len(self.image_paths)

    def read_path2list(self, path):
        image_files = sorted(os.listdir(path))
        return [os.path.join(path, f) for f in image_files]

    def normalize_image(self, image, clip_range=(0, 1)):
        image = image.astype(np.float32)
        image /= 255
        image = np.clip(image, clip_range[0], clip_range[1])
        return image

    def _get_label_unique(self):
        return np.unique(self.label_trans)

if __name__=="__main__":
    image_path = r"D:\PythonProject\pytorch_segmentation_3D\DSCNet_3D\SegDataset\MRI_Hippocampus\imagesTe"
    label_path = r"D:\PythonProject\pytorch_segmentation_3D\DSCNet_3D\SegDataset\MRI_Hippocampus\labelsTe"

    dataset = NiftiDataset3D(image_path, label_path, num_classes=3)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for i, (image, onehotlabel) in enumerate(dataset):
        print(f"Sample {i+1}:")
        print("Image shape:", image.shape)
        print("One Hot Label shape:", onehotlabel.shape)
        print(np.unique(onehotlabel), np.unique(onehotlabel[0]), np.unique(onehotlabel[1]),
              np.unique(onehotlabel[2]))
        print(dataset._get_label_unique())
        print("-------------------")
