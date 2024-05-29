import os
from pathlib import Path
from os import getcwd

IMAGE_EXTENSIONS = ('png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'tif', 'raw')
IMAGE_EXTENSIONS_SET = set(IMAGE_EXTENSIONS)

def getPhotopath(paths, cd=False, debug=False):
    """
    :param paths: 文件夹路径
    :param cd: 添加当前运行的路径名,这是使用了相对路径才能用的
    :param debug: 开启打印文件名错误的名字
    :return: 包含图片路径的列表
    """
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
        if file_ext[1:] in IMAGE_EXTENSIONS:
            imgfile.append(newph)
    if cd:
        # 使用了相对路径, 使用这个补全
        cdd = getcwd()
        imgfile = [os.path.join(cdd, file).replace("\\", "/") for file in imgfile]
        allfile = [os.path.join(cdd, file).replace("\\", "/") for file in allfile]
    return imgfile, allfile


def SearchFilePath(filedir, format='png'):
    """What is returned is a list that includes all paths under the target path that match the suffix."""
    search_file_path = []
    for root, dirs, files in os.walk(filedir):
        for filespath in files:
            if str(filespath).endswith(format):
                search_file_path.append(os.path.join(root, filespath))
    return search_file_path


def split_path2list(path_str):
    """
    path_list = split_path2list('D:\PythonProject\MB_TaylorFormer\DehazeFormer\data\rshazy\test\GT\220.png')
    Return:
        ['D:\\', 'PythonProject', 'MB_TaylorFormer', 'DehazeFormer', 'data', 'rshazy', 'test', 'GT', '220.png']
    """
    path = Path(path_str)
    path_parts = path.parts
    return list(path_parts)

if __name__=="__main__":
    filedir = r'D:\PythonProject\MB_TaylorFormer\DehazeFormer\data\rshazy\test\GT\220.png'
    file_list = SearchFilePath(filedir)

    for file_path in file_list:
        print(file_path)

    path_list = split_path2list(filedir)
    print(path_list)