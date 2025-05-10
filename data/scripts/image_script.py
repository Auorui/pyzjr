import os
import shutil
import math
from PIL import Image
from pyzjr.data.utils.tuplefun import to_2tuple
from pyzjr.data.utils.path import get_image_path

def modify_images_size(target_path, target_hwsize, save_path=None, message=True):
    """批量修改图像大小"""
    h, w = to_2tuple(target_hwsize)
    images_paths = get_image_path(target_path)
    os.makedirs(save_path, exist_ok=True)
    for i, one_image_path in enumerate(images_paths):
        try:
            image = Image.open(one_image_path)
            resized_image = image.resize((w, h))
            base_name = os.path.basename(one_image_path)
            if save_path is not None:
                new_path = os.path.join(save_path, base_name)
            else:
                new_path = one_image_path

            resized_image.save(new_path)
            if message:
                print(f"Resized {one_image_path} to {new_path}")
        except Exception as e:
            print(f"Error resizing {one_image_path}: {e}")

def split_train_val_txt(target_path, train_ratio=.8, val_ratio=.2, onlybasename=False):
    """
    如果 train_ratio + val_ratio = 1 表示只划分训练集和验证集, train_ratio + val_ratio < 1
    表示将剩余的比例划分为测试集
    """
    assert train_ratio + val_ratio <= 1
    test_ratio = 1. - (train_ratio + val_ratio)
    images_paths = get_image_path(target_path)

    num_images = len(images_paths)
    num_train = round(num_images * train_ratio)
    num_val = num_images - num_train if test_ratio == 0 else math.ceil(num_images * val_ratio)
    num_test = 0 if test_ratio == 0 else num_images - (num_train + num_val)

    with open(os.path.join(target_path, 'train.txt'), 'w') as train_file, \
            open(os.path.join(target_path, 'val.txt'), 'w') as val_file, \
            open(os.path.join(target_path, 'test.txt'), 'w') as test_file:
        for i, image_path in enumerate(images_paths):
            if onlybasename:
                image_name, _ = os.path.splitext(os.path.basename(image_path))
            else:
                image_name = image_path
            if i < num_train:
                train_file.write(f"{image_name}\n")
            elif i < num_train + num_val:
                val_file.write(f"{image_name}\n")
            else:
                test_file.write(f"{image_name}\n")

    print(f"Successfully split {num_images} images into {num_train} train, {num_val} val, and {num_test} test.")

def copy_images_to_directory(target_path, save_folder, message=True):
    """复制整个文件夹（图像）到另外一个文件夹"""
    try:
        os.makedirs(save_folder, exist_ok=True)
        source_path = get_image_path(target_path)
        for img_path in source_path:
            base_file_name = os.path.basename(img_path)
            destination_path = os.path.join(save_folder, base_file_name)
            shutil.copy2(img_path, destination_path)
            if message:
                print(f"Successfully copied folder: {img_path} to {save_folder}")
    except Exception as e:
        print(f"Error copying folder, {e}")


if __name__=="__main__":
    target_path = ...
    save_path = ...
    # modify_images_suffix(target_path, 'png')
    modify_images_size(target_path, (420, 512), './test2')
    # copy_images_to_directory(target_path, save_path)
    # split_train_val_txt(target_path, val_ratio=.2,onlybasename=True)
    # batch_rename_images(target_path, save_path, start_number=None)