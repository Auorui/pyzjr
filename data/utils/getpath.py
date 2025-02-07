import os
from pathlib import Path

import pyzjr.Z as Z
from pyzjr.data.base import natsorted

def getPhotoPath(path, debug=False):
    """
    :param path: 文件夹路径
    :param debug: 开启打印文件名错误的名字
    :return: 包含图片路径的列表
    """
    imgfile = []
    allfile = []
    file_list = os.listdir(path)
    for i in file_list:
        if debug:
            if i[0] in ['n', 't', 'r', 'b', 'f'] or i[0].isdigit():
                print(f"File name error occurred at the beginning of {i}!")
        newph = os.path.join(path, i).replace("\\", "/")
        allfile.append(newph)
        _, file_ext = os.path.splitext(newph)
        if file_ext[1:] in Z.IMG_FORMATS:
            imgfile.append(newph)

    return natsorted(imgfile), natsorted(allfile)

def SearchFilePath(filedir, file_ext='.png'):
    """What is returned is a list that includes all paths under the target path that match the suffix."""
    search_file_path = []
    for root, dirs, files in os.walk(filedir):
        for filespath in files:
            if str(filespath).endswith(file_ext):
                search_file_path.append(os.path.join(root, filespath))
    return natsorted(search_file_path)

def SearchFileName(target_path, file_ext='.png'):
    """Only search for file names in the appropriate format in the target folder"""
    all_files = os.listdir(target_path)
    png_files = [file for file in all_files if file.lower().endswith(file_ext)]
    sorted_png_files = natsorted(png_files)
    return sorted_png_files

def split_path2list(path_str):
    """
    path_list = split_path2list('D:\PythonProject\MB_TaylorFormer\DehazeFormer\data\rshazy\test\GT\220.png')
    Return:
        ['D:\\', 'PythonProject', 'MB_TaylorFormer', 'DehazeFormer', 'data', 'rshazy', 'test', 'GT', '220.png']
    """
    path = Path(path_str)
    path_parts = path.parts
    return list(path_parts)

class AbsPathOps():
    def __init__(self, path):
        self.path_list = split_path2list(path)
        self._full_path = path

    @property
    def basename(self):
        return self.path_list[-1]

    @property
    def format(self):
        basename = self.basename
        if "." in basename:
            return basename.split(".")[-1]
        else:
            return ""

    @property
    def disk(self):
        if not os.path.isabs(path):
            raise ValueError(f"The provided path {path} is not an absolute path.")
        return self.path_list[0]

    @property
    def dirname(self):
        return os.sep.join(self.path_list[:-1])

    @property
    def filename(self):
        basename = self.basename
        if "." in basename:
            return basename.rsplit(".", 1)[0]  # 使用 rsplit 从右侧开始分割，并限制为分割一次
        else:
            return basename

    @property
    def parent(self):
        return os.path.dirname(self.dirname)


if __name__ == "__main__":
    path = r"D:\PythonProject\pyzjrPyPi\pyzjr\data"
    print(SearchFilePath(path))
    pathops = AbsPathOps(path)
    print(pathops.basename)
    print(pathops.format)
    print(pathops.disk)
    print(pathops.dirname)
    print(pathops.filename)
    print(pathops.parent)