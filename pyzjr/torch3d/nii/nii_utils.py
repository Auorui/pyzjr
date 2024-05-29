import os
import glob
import nibabel as nib
import numpy as np

def get_nii_path(folder_path):
    """
    在给定的文件夹路径中搜索并返回所有.nii.gz文件的路径。
    Args:
        folder_path (str): 要搜索的文件夹路径。

    Return:
        list: 包含所有找到的.nii.gz文件路径的列表。
    """
    image_files = []
    for extension in ['*.nii.gz']:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))
    return image_files

def nii_load(path):
    """
    读取NIfTI格式的图像文件。 ---> (w, h, d) [ x, y, z ]
    Args:
        path: str, NIfTI图像文件的路径

    Return:
        data: numpy.ndarray, 包含图像数据的三维数组。
        affine: numpy.ndarray, 图像的仿射矩阵，用于将图像坐标转换为世界坐标。
        header: nibabel.Nifti1Header, NIfTI图像的头文件信息。
    """
    nii_image = nib.load(path)
    data = nii_image.get_fdata()
    affine = nii_image.affine
    header = nii_image.header
    return data, affine, header

def nii_write(outputs_np, affine, header, filename_out):
    """
    将给定的 Numpy 数组保存为 NIfTI 格式的图像文件。

    Args:
        outputs_np: numpy.ndarray, 要保存的图像数据。
        affine: numpy.ndarray, 图像的仿射矩阵，描述了图像的空间位置和定向信息。
        header: numpy.ndarray, 图像的头部信息
        filename_out: str, 输出文件的路径，包括文件名及其扩展名（如'.nii.gz'）。
    """
    outputs_nib = nib.Nifti1Image(outputs_np, affine, header)
    outputs_nib.header['qform_code'] = 1
    outputs_nib.header['sform_code'] = 0
    outputs_nib.to_filename(filename_out)

def center_crop_nii(data, crop_shape):
    """
    对 NIfTI 图像进行中心裁剪。
    Args:
        data: numpy.ndarray, NIfTI 图像数据。
        crop_shape: tuple, 裁剪后的图像形状 (depth, height, width)。

    Returns:
        cropped_data: numpy.ndarray, 裁剪后的图像数据。
    """
    start_x = (data.shape[2] - crop_shape[2]) // 2
    start_y = (data.shape[1] - crop_shape[1]) // 2
    start_z = (data.shape[0] - crop_shape[0]) // 2
    cropped_data = data[start_z:start_z + crop_shape[0],
                   start_y:start_y + crop_shape[1],
                   start_x:start_x + crop_shape[2]]
    return cropped_data.astype(np.uint8)


def center_scale_nii(data, scale_factor):
    """
    对给定的NIfTI格式数据进行中心缩放。

    参数：
        data (numpy.ndarray)：NIfTI数据的Numpy数组表示。
        scale_factor (float)：缩放因子，用于指定缩放的比例。

    返回：
        numpy.ndarray：经过中心缩放后的数据。

    注意：
        这个函数假设输入的数据是3D的，并且是一个Numpy数组。你应该调用 nii_load 读取
        缩放因子应为正数，表示缩小（<1）或放大（>1）的比例。
    """
    new_shape = np.round(np.array(data.shape) * scale_factor).astype(int)
    offset = (np.array(data.shape) - new_shape) // 2
    scaled_data = data[offset[0]:offset[0]+new_shape[0],
                  offset[1]:offset[1]+new_shape[1],
                  offset[2]:offset[2]+new_shape[2]]
    return scaled_data.astype(np.uint8)