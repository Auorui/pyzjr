import cv2
import os
import imghdr
from os import getcwd
import shutil
import tqdm

def getPhotopath(paths,cd=False,debug=True):
    """
    * log
        0.0.19以后修改了一个比较大的bug
        1.0.2后将图片和所有文件路径分开
        1.0.5功能全部完善,不会再进行更新
    :param paths: 文件夹路径
    :param cd:添加当前运行的路径名
    :param debug:开启打印文件名错误的名字
    :return: 包含图片路径的列表
    """
    img_formats = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tif', 'tiff', 'webp', 'raw']
    imgfile = []
    allfile = []
    file_list = os.listdir(paths)
    for i in file_list:
        if debug:
            if i[0] in ['n', 't', 'r', 'b', 'f'] or i[0].isdigit():
                print(f"[pyzjr]: File name:Error occurred at the beginning of {i}!")
        newph = os.path.join(paths, i).replace("\\", "/")
        allfile.append(newph)
        _, file_ext = os.path.splitext(newph)
        if file_ext[1:] in img_formats:
            imgfile.append(newph)
    if cd:
        cdd = getcwd()
        imgfile = [os.path.join(cdd, file).replace("\\", "/") for file in imgfile]
        allfile = [os.path.join(cdd, file).replace("\\", "/") for file in allfile]
    return imgfile,allfile

def RenameFile(image_folder_path, save_image_folder_path, newbasename='re', type=3, format=None):
    """
    对图像文件夹进行重命名
    :param image_folder_path: 输入图像的路径
    :param save_image_folder_path: 保存图像的路径
    :param newbasename: 基础名
    :param type: 默认为3——001、002、003、004...
    :param format: 扩展名,如png、jpg,默认用原本的扩展名
    :return:
    """
    savepath = CreateFolder(save_image_folder_path,debug=False)
    imglist, allist = getPhotopath(image_folder_path, debug=False)

    total_files = len(imglist)
    for i, file in tqdm.tqdm(enumerate(imglist), total=total_files, desc='Renaming files'):
        properties = os.path.basename(file)
        name, ext = os.path.splitext(properties)
        padded_i = str(i).zfill(type)
        if format is not None:
            newname = f"{newbasename}{padded_i}."+format
        else:
            newname = f"{newbasename}{padded_i}.{ext[1:]}"
        new_path = os.path.join(savepath, newname)
        shutil.copy(file, new_path)
    print("[pyzjr]:Batch renaming and saving of files completed!")

def CreateFolder(folder_path,debug=True):
    """确保文件夹存在"""
    if not os.path.exists(folder_path):
        try:
            os.mkdir(folder_path)
            if debug:
                print(f"[pyzjr]:Folder_{folder_path} created successfully!")
        except OSError:
            if debug:
                print(f"[pyzjr]:Folder_{folder_path} creation failed!")
    else:
        if debug:
            print(f"[pyzjr]:Folder_{folder_path} already exists!")
    return folder_path

def ImageAttribute(image):
    """获取图片属性"""
    properties = {}
    if isinstance(image, str):  # 如果传入的是文件路径
        properties['name'] = os.path.basename(image)
        properties['format'] = imghdr.what(image)
        properties['fsize'] = os.path.getsize(image)
        image = cv2.imread(image)
    else:  # 如果传入的是图片数据
        properties['name'] = "Nan"
        properties['format'] = "Nan"
        properties['fsize'] = image.nbytes
    properties["shape"] = image.shape
    properties["dtype"] = image.dtype
    properties['size'] = image.size
    return properties
