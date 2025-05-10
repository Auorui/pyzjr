import os
import shutil
from pyzjr.data.utils.path import get_image_path

def batch_rename_images(
        target_path,
        save_path,
        start_index=None,
        prefix=None,
        suffix=None,
        format=None,
        num_type=1,
        message=True
):
    """
    重命名图像文件夹中的所有图像文件并保存到指定文件夹
    :param target_path: 目标文件路径
    :param save_path: 文件夹的保存路径
    :param start_index: 默认为 1, 从多少号开始
    :param prefix: 重命名的通用格式前缀, 如 rename001.png, rename002.png...
    :param suffix: 重命名的通用格式后缀, 如 001rename.png, 002rename.png...
    :param format (str): 新的后缀名，不需要包含点（.）
    :param num_type: 数字长度, 比如 3 表示 005
    :param message: 是否打印修改信息
    """
    os.makedirs(save_path, exist_ok=True)
    images_paths = get_image_path(target_path)
    current_num = start_index if start_index is not None else 1

    for i, image_path in enumerate(images_paths):
        image_name = os.path.basename(image_path)
        name, ext = os.path.splitext(image_name)
        if format is None:
            ext = ext
        else:
            ext = f'.{format}'
        padded_i = str(current_num).zfill(num_type)
        if prefix and suffix:
            new_image_name = f"{prefix}{padded_i}{suffix}{ext}"
        elif prefix:
            new_image_name = f"{prefix}{padded_i}{ext}"
        elif suffix:
            new_image_name = f"{padded_i}{suffix}{ext}"
        else:
            new_image_name = f"{padded_i}{ext}"

        new_path = os.path.join(save_path, new_image_name)
        current_num += 1
        if message:
            print(f"{i + 1} Successfully rename {image_path} to {new_path}")
        shutil.copy(image_path, new_path)

    print("Batch renaming and saving of files completed!")

def convert_suffix(file_path, format):
    """
    将文件路径中的后缀名转换为新的后缀名

    Args:
        file_path (str): 要转换后缀名的文件路径。
        format (str): 新的后缀名，不需要包含点（.）。

    Returns:
        str: 转换后的文件路径。
    """
    base_name, ext = os.path.splitext(file_path)
    new_file_path = base_name + '.' + format
    return new_file_path

def modify_images_suffix(target_path, format='png', message=True):
    """批量修改图像文件后缀"""
    images_paths = get_image_path(target_path)
    for i, one_image_path in enumerate(images_paths):
        new_path = convert_suffix(one_image_path, format)
        os.rename(one_image_path, new_path)
        if message:
            print(f"Converting {one_image_path} to {new_path}")


if __name__=="__main__":
    target_path = r'E:\PythonProject\Pytorch_Segmentation_Auorui\data_utils\SateHaze1K\Haze1k_moderate\val\hazy'
    save_path = 'SateHaze1K/Haze1k_moderate/val/hazy'

    batch_rename_images(target_path, save_path, start_index=321, num_type=1)
    # modify_images_suffix(save_path, format='jpg')