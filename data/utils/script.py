import os
import shutil
from PIL import Image
from pyzjr.data.utils.path import get_image_path

def norm_to_abs(cx_norm, cy_norm, w_norm, h_norm, img_w, img_h):
    """
    Convert normalized bounding box coordinates to absolute pixel coordinates.

    This function transforms bounding box parameters from normalized [0,1] range
    (common in machine learning datasets) to absolute pixel values based on image dimensions.

    :param cx_norm: Normalized x-coordinate of bounding box center [0,1]
    :param cy_norm: Normalized y-coordinate of bounding box center [0,1]
    :param w_norm: Normalized width of bounding box [0,1]
    :param h_norm: Normalized height of bounding box [0,1]
    :param img_w: Width of the image in pixels
    :param img_h: Height of the image in pixels
    :return: list: [x_min, y_min, x_max, y_max] absolute coordinates (pixels) representing:
                                                x_min: Left boundary of bounding box
                                                y_min: Top boundary of bounding box
                                                x_max: Right boundary of bounding box
                                                y_max: Bottom boundary of bounding box
    """
    cx = cx_norm * img_w
    cy = cy_norm * img_h
    w = w_norm * img_w
    h = h_norm * img_h
    x_min = cx - w/2
    y_min = cy - h/2
    x_max = cx + w/2
    y_max = cy + h/2
    return [x_min, y_min, x_max, y_max]


def abs_to_norm(x_min, y_min, x_max, y_max, img_w, img_h):
    """
    Convert absolute pixel coordinates to normalized bounding box coordinates.

    This function transforms bounding box parameters from absolute pixel values
    to normalized [0,1] range (common in machine learning datasets).

    :param x_min: Left boundary of bounding box in pixels
    :param y_min: Top boundary of bounding box in pixels
    :param x_max: Right boundary of bounding box in pixels
    :param y_max: Bottom boundary of bounding box in pixels
    :param img_w: Width of the image in pixels
    :param img_h: Height of the image in pixels
    :return: list: [cx_norm, cy_norm, w_norm, h_norm] normalized coordinates representing:
                    cx_norm: Normalized x-coordinate of bounding box center [0,1]
                    cy_norm: Normalized y-coordinate of bounding box center [0,1]
                    w_norm: Normalized width of bounding box [0,1]
                    h_norm: Normalized height of bounding box [0,1]
    """
    w = x_max - x_min
    h = y_max - y_min
    cx = x_min + w / 2
    cy = y_min + h / 2
    cx_norm = cx / img_w
    cy_norm = cy / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return [cx_norm, cy_norm, w_norm, h_norm]

def batch_modify_images(
        target_dir,
        save_dir,
        start_index=None,
        prefix='',
        suffix='',
        format=None,
        target_shape=None,
        num_type=1,
):
    """
    重命名图像文件夹中的所有图像文件并保存到指定文件夹
    :param target_dir: 目标文件路径
    :param save_dir: 文件夹的保存路径
    :param start_index: 默认为 1, 从多少号开始
    :param prefix: 重命名的通用格式前缀, 如 rename001.png, rename002.png...
    :param suffix: 重命名的通用格式后缀, 如 001rename.png, 002rename.png...
    :param format (str): 新的后缀名，不需要包含点（.）
    :param num_type: 数字长度, 比如 3 表示 005
    """
    os.makedirs(save_dir, exist_ok=True)
    images_paths = get_image_path(target_dir)
    total_nums = len(images_paths)
    for i, image_path in enumerate(images_paths):
        image_name = os.path.basename(image_path)
        name, ext = os.path.splitext(image_name)
        re_ext = f".{format}" if format is not None else ext
        if start_index:
            padded_i = str(start_index).zfill(num_type)
            start_index += 1
        else:
            padded_i = name
        new_image_name = f"{prefix}{padded_i}{suffix}{re_ext}"
        new_path = os.path.join(save_dir, new_image_name)
        image = Image.open(image_path)
        if image.mode != 'RGB' and image.mode != 'RGBA':
            image = image.convert('RGBA')
        if target_shape:
            height, width = target_shape
            image = image.resize((width, height))
        if format and format.lower() == 'png':
            image.save(new_path, 'PNG')
        elif format and format.lower() in ['jpg', 'jpeg']:
            image = image.convert('RGB')
            image.save(new_path, 'JPEG', quality=100)
        else:
            image.save(new_path)
        print(f"{i + 1}/{total_nums} Successfully rename {image_path} to {new_path}")

def copy_files(file_list, target_dir):
    """
    将多个文件复制到指定保存文件夹

    Args:
        file_list (list): 需要复制的文件路径列表（如 [path1, path2, ...]）
        target_dir (str): 目标保存文件夹路径
    """
    try:
        os.makedirs(target_dir, exist_ok=True)
        for file_path in file_list:
            base_file_name = os.path.basename(file_path)
            destination_path = os.path.join(target_dir, base_file_name)
            shutil.copy2(file_path, destination_path)
            print(f"Successfully copied: {file_path} -> {target_dir}")
    except Exception as e:
        print(f"Error copying files: {e}")


def move_files(file_list, target_dir):
    """
    将多个文件移动到指定目录（保留元数据，自动创建目标目录）

    Args:
        file_list (list): 需要移动的文件路径列表（如 [path1, path2, ...]）
        target_dir (str): 目标目录路径
    """
    try:
        os.makedirs(target_dir, exist_ok=True)
        for file_path in file_list:
            base_name = os.path.basename(file_path)
            dest_path = os.path.join(target_dir, base_name)
            shutil.move(file_path, dest_path)
            print(f"Successfully moved: {file_path} -> {dest_path}")
    except Exception as e:
        print(f"Error moving files: {str(e)}")