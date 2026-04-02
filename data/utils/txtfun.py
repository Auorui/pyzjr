import os
import math
import random
import shutil
from pyzjr.data.utils.listfun import natsorted
from pyzjr.utils.check import is_str, is_list_or_tuple
from pyzjr.data.utils.path import get_image_path

def generate_txt(dir_path, onlybasename=False, txt_path=r'./output.txt'):
    """将指定文件夹下的文件路径（或文件名）写入到指定的文本文件中"""
    f = open(txt_path, "w")
    files = natsorted(os.listdir(dir_path))
    files_num = len(files)
    index_count = 0
    count = 0
    for file in files:
        index_count = index_count + 1
        path = os.path.splitext(file)[0] if onlybasename else os.path.join(dir_path, str(file))
        if count == files_num - 1:
            f.write(path)
            break
        if index_count >= 0:
            f.write(path + "\n")
            count = count + 1
    f.close()

def read_txt(txt_path: str):
    """Read lines from a text file into a list."""
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def write_txt(file_path, content, encoding='utf-8'):
    """将指定的内容写入到指定的文本文件中"""
    if os.path.exists(file_path):
        os.remove(file_path)

    if is_list_or_tuple(content):
        with open(file_path, 'a', encoding=encoding) as file:
            for i in content:
                if is_str(i):
                    file.write(i)
                else:
                    file.write(str(i))
                file.write("\n")

    elif is_str(content):
        with open(file_path, 'w', encoding=encoding) as file:
            file.write(content)

def split_train_val_txt(dir_path, train_ratio=.8, val_ratio=.2, onlybasename=False,
                        shuffle=True, save_in_current_dir=True, seed=0):
    """
    如果 train_ratio + val_ratio = 1 表示只划分训练集和验证集, train_ratio + val_ratio < 1
    表示将剩余的比例划分为测试集
    """
    assert train_ratio + val_ratio <= 1
    test_ratio = 1. - (train_ratio + val_ratio)
    images_paths = get_image_path(dir_path)

    if shuffle:
        seed_number = int(seed)
        random.seed(seed_number)
        random.shuffle(images_paths)

    num_images = len(images_paths)
    num_train = round(num_images * train_ratio)
    num_val = num_images - num_train if test_ratio == 0 else math.ceil(num_images * val_ratio)
    num_test = 0 if test_ratio == 0 else num_images - (num_train + num_val)

    save_dir = os.getcwd() if save_in_current_dir else dir_path

    train_path = os.path.join(save_dir, 'train.txt')
    val_path = os.path.join(save_dir, 'val.txt')
    test_path = os.path.join(save_dir, 'test.txt')

    with open(train_path, 'w') as train_file, \
            open(val_path, 'w') as val_file, \
            open(test_path, 'w') as test_file:

        for i, image_path in enumerate(images_paths):
            if onlybasename:
                image_name = os.path.splitext(os.path.basename(image_path))[0]
            else:
                image_name = image_path

            if i < num_train:
                train_file.write(f"{image_name}\n")
            elif i < num_train + num_val:
                val_file.write(f"{image_name}\n")
            else:
                test_file.write(f"{image_name}\n")

    print(f"Successfully split {num_images} images into: "
          f"{num_train} train, {num_val} val, {num_test} test")
    print(f"Files saved in: {save_dir}")

def copy_images_from_txt(txt_path, target_dir, create_subdirs=False):
    """
    根据txt文件中的路径复制图像到目标目录

    参数:
        txt_path: 包含图像路径的txt文件
        target_dir: 目标目录（图像将被复制到这里）
        create_subdirs: 是否在目标目录中创建与原路径相同的子目录结构
    """
    os.makedirs(target_dir, exist_ok=True)
    with open(txt_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    copied_count = 0
    for src_path in image_paths:
        if not os.path.exists(src_path):
            print(f"Warning: Source file does not exist - {src_path}")
            continue
        filename = os.path.basename(src_path)
        if create_subdirs:
            rel_path = os.path.relpath(os.path.dirname(src_path), os.path.dirname(txt_path))
            dest_path = os.path.join(target_dir, rel_path, filename)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        else:
            dest_path = os.path.join(target_dir, filename)

        shutil.copy2(src_path, dest_path)
        copied_count += 1

    print(f"Successfully copied {copied_count}/{len(image_paths)} files to {target_dir}")
    return copied_count

def read_detection_labels_txt(file_path):
    """
    Read object detection labels from a text file in normalized coordinate format.
    Format per line: <class_id> <center_x_norm> <center_y_norm> <width_norm> <height_norm>

    Args:
        file_path (str): Path to the annotation text file

    Returns:
        list: List of annotations where each annotation is a tuple:
              (class_id, cx_norm, cy_norm, width_norm, height_norm)

    Example:
        Input file content:
            2 0.64375 0.5983333333333334 0.26 0.3466666666666667
            2 0 0.415 0.2475 0.33

        Output:
            [(2, 0.64375, 0.5983333333333334, 0.26, 0.3466666666666667),
             (2, 0, 0.415, 0.2475, 0.33)]
    """
    annotations = []

    try:
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, 1):
                # Skip empty lines and comment lines
                stripped_line = line.strip()
                if not stripped_line or stripped_line.startswith('#'):
                    continue

                # Split line into components
                parts = stripped_line.split()

                # Validate number of components
                if len(parts) != 5:
                    raise ValueError(f"Invalid format at line {line_number}: "
                                     f"Expected 5 values, found {len(parts)}")

                try:
                    # Parse values with type conversion
                    class_id = int(parts[0])
                    cx_norm = float(parts[1])
                    cy_norm = float(parts[2])
                    width_norm = float(parts[3])
                    height_norm = float(parts[4])

                    # Validate normalized coordinate ranges
                    for i, val in enumerate([cx_norm, cy_norm, width_norm, height_norm], start=1):
                        if val < 0 or val > 1:
                            print(f"Warning: Value at line {line_number} position {i} "
                                  f"({val}) is outside [0,1] range")

                    annotations.append((class_id, cx_norm, cy_norm, width_norm, height_norm))

                except ValueError as e:
                    print(f"Error parsing line {line_number}: {e}")
                    continue

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Unexpected error reading file: {e}")
        return []

    return annotations

if __name__ == "__main__":
    file_dir = r"D:\PythonProject\pyzjrPyPi\pyzjr\data"
    generate_txt(file_dir, onlybasename=True)
    print(read_txt("./output.txt"))