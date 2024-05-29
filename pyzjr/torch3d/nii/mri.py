"""
对 MRI（磁共振成像）的海马体区域分割数据集进行一系列的基础操作
"""
import os
import cv2
import numpy as np
import nibabel as nib

Hippocampal_label_categories = {
    0: 0,  # 黑色  --->  背景
    1: 128,  # 灰色  --->  右海马体区域
    2: 255  # 白色  --->  左海马体区域
}    # 映射是通过 np.unique 函数查找的

def save_slice_as_label(slice_data, output_dir, slice_index, label_to_gray, prefix='slice', file_format='png'):
    """
    保存单个切片为图片文件。(面对标签进行操作)

    Args:
        slice_data: numpy.ndarray, 切片数据。
        output_dir: str, 输出目录的路径。
        slice_index: int, 切片的索引。
        prefix: str, 输出文件名的前缀。
        file_format: str, 图片文件的格式（如'png', 'jpg'等）。
    """
    output_path = os.path.join(output_dir, f"{prefix}_{slice_index}.{file_format}")
    slice_data_gray = np.vectorize(lambda x: label_to_gray[x])(slice_data)
    cv2.imwrite(output_path, slice_data_gray)

def save_slice_as_image(slice_data, output_dir, slice_index, prefix='slice', file_format='png'):
    """
    保存单个切片为图片文件。

    Args:
        slice_data: numpy.ndarray, 切片数据。
        output_dir: str, 输出目录的路径。
        slice_index: int, 切片的索引。
        prefix: str, 输出文件名的前缀。
        file_format: str, 图片文件的格式（如'png', 'jpg'等）。
    """
    output_path = os.path.join(output_dir, f"{prefix}_{slice_index}.{file_format}")
    slice_data_normalized = (
            (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255).astype(np.uint8)
    cv2.imwrite(output_path, slice_data_normalized)

def save_slices_along_axis(image_data, output_dir, axis, is_label=None, prefix='slice', file_format='png'):
    """
    沿着指定的轴保存NIfTI图像的切片。

    Args:
        image_data: numpy.ndarray, NIfTI图像的数据。
        output_dir: str, 输出目录的路径。
        axis: int, 切片的轴方向（0代表X轴，1代表Y轴，2代表Z轴）.
        is_label: 是否对标签进行操作, Hippocampal_label_categories, etc.
        prefix: str, 输出文件名的前缀。
        file_format: str, 图片文件的格式（如'png', 'jpg'等）.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if axis == 0:  # X轴
        for slice_index in range(image_data.shape[0]):
            slice_data = image_data[slice_index, :, :]
            if is_label:
                save_slice_as_label(slice_data, output_dir, slice_index, is_label, prefix, file_format)
            else:
                save_slice_as_image(slice_data, output_dir, slice_index, prefix, file_format)
    elif axis == 1:  # Y轴
        for slice_index in range(image_data.shape[1]):
            slice_data = image_data[:, slice_index, :]
            if is_label:
                save_slice_as_label(slice_data, output_dir, slice_index, is_label, prefix, file_format)
            else:
                save_slice_as_image(slice_data, output_dir, slice_index, prefix, file_format)
    elif axis == 2:  # Z轴
        for slice_index in range(image_data.shape[2]):
            slice_data = image_data[:, :, slice_index]
            if is_label:
                save_slice_as_label(slice_data, output_dir, slice_index, is_label, prefix, file_format)
            else:
                save_slice_as_image(slice_data, output_dir, slice_index, prefix, file_format)
    else:
        raise ValueError("Invalid axis. Axis should be 0 (X), 1 (Y), or 2 (Z).")


if __name__=="__main__":
    img_path = "xxx.nii.gz"
    image_data = nib.load(img_path)
    image_data = image_data.get_fdata()
    affine = image_data.affine
    header = image_data.header
    output_dir = ...

    output_dirs_x = os.path.join(output_dir, "x")
    output_dirs_y = os.path.join(output_dir, "y")
    output_dirs_z = os.path.join(output_dir, "z")
    save_slices_along_axis(image_data, output_dirs_x, axis=0, is_label=Hippocampal_label_categories)
    save_slices_along_axis(image_data, output_dirs_y, axis=1, is_label=Hippocampal_label_categories)
    save_slices_along_axis(image_data, output_dirs_z, axis=2, is_label=Hippocampal_label_categories)
