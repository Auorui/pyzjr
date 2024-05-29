import os
from PIL import Image
from pyzjr.data.basedataset import BaseDataset

__all__ = ["BaseClsDataset", "VOCSegmentation"]


class BaseClsDataset(BaseDataset):

    def __init__(self, file_txt, image_transform=None):
        """
        Supports txt file paths that match the format, or have been processed into the format list below.
        file_txt format: [path/to/xxx.jpg 0,
                          path/to/xxx.jpg 1,           OR           path/to/xxx.txt
                          path/to/xxx.jpg 2,
                          ...]
        """
        super().__init__()
        file_txt_result = self.load_txt_path_or_list(file_txt)
        if self._check_file_txt_format(file_txt_result):
            self.file_txt = file_txt_result
        self.image_transform = image_transform

    def __len__(self):
        return len(self.file_txt)

    def _check_file_txt_format(self, file_txt):
        for line in file_txt:
            parts = line.split()
            assert len(parts) == 2, f"BaseClsDataset: Invalid format in line: {line}"
            file_path, digit_label = parts
            assert os.path.exists(file_path), f"BaseClsDataset: File not found: {file_path}"
            assert digit_label.isdigit(), f"BaseClsDataset: Invalid digit label: {digit_label}"
        return True

    def __getitem__(self, idx):
        print(self.file_txt[idx])
        file_path, digit_label = self.file_txt[idx].split()
        raw_image = Image.open(file_path)
        image = raw_image.convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        return image, int(digit_label)

class VOCSegmentation(BaseDataset):
    def __init__(
            self,
            root,
            year="2012",
            image_set="train",
            transforms=None,
    ):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2008", "2009", "2010", "2011", "2012"], \
            "year can only choose 2007 to 2012"
        if year == "2007" and image_set == "test":
            year = "2007-test"
        self.year = year
        self.transforms = transforms
        voc_root = os.path.join(root, f"VOC{year}")
        txt_name = image_set + ".txt"

        assert os.path.exists(voc_root), "path '{}' does not exist.".format(voc_root)

        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')
        txt_path = os.path.join(voc_root, "ImageSets", "Segmentation", txt_name)

        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)

        file_names = self.load_txt_path_or_list(txt_path)
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index]).convert('L')

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)





# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#
#     dataset_path = r"D:\PythonProject\uhrnet\data\pneumonia"
#     train_dataset = SegmentTwoDataset(dataset_path, train=True)
#     val_dataset = SegmentTwoDataset(dataset_path, train=False)
#
#     train_loader = DataLoader(train_dataset, shuffle=True, batch_size=2, num_workers=8, pin_memory=True,
#                               drop_last=True)
#     val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1, num_workers=4, pin_memory=True,
#                             drop_last=True)
#
#     for im, label, seglabel in train_loader:
#         print(im.shape, label.shape, seglabel.shape)