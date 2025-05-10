import os
from pathlib import Path
from pyzjr.data.utils.listfun import natsorted

def get_image_path(path):
    """
    获取当前文件夹下符合图像格式的图像路径，并修正了转义字符的问题
    详见: https://blog.csdn.net/m0_62919535/article/details/132199978
    """
    imgfile = []
    file_list = os.listdir(path)
    for i in file_list:
        new_path = os.path.join(path, i).replace("\\", "/")
        _, file_ext = os.path.splitext(new_path)
        if file_ext[1:] in ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'):
            imgfile.append(new_path)
    return natsorted(imgfile)

def DeepSearchFilePath(target_path, file_ext='.png'):
    """
    深度搜索当前文件夹下包括其下文件夹当中符合后缀格式的文件路径, 为 SearchFilePath 的改进版本
    :param target_path: 目标的文件夹路径
    :param file_ext: 还有 ’.‘ 的后缀格式
    """
    search_file_path = []
    for root, dirs, files in os.walk(target_path):
        for filespath in files:
            if str(filespath).endswith(file_ext):
                search_file_path.append(os.path.join(root, filespath))
    return natsorted(search_file_path)

def SearchFilePath(target_path, file_ext='.png'):
    """
    仅获取目标文件夹下面符合后缀格式的文件路径
    :param target_path: 目标的文件夹路径
    :param file_ext: 还有 ’.‘ 的后缀格式
    """
    search_file_path = []
    files = os.listdir(target_path)
    for filespath in files:
        if str(filespath).endswith(file_ext):
            search_file_path.append(os.path.join(target_path, filespath))
    return natsorted(search_file_path)

def SearchFileName(target_path, file_ext='.png'):
    """
    仅获取目标文件夹下面符合后缀格式的文件名
    :param target_path: 目标的文件夹路径
    :param file_ext: 还有 ’.‘ 的后缀格式
    """
    all_files = os.listdir(target_path)
    png_files = [file for file in all_files if file.lower().endswith(file_ext)]
    return natsorted(png_files)

def split_path(path):
    """
    根据斜杠分解路径成 list
    path_list = split_path2list('D:\PythonProject\MB_TaylorFormer\DehazeFormer\data\rshazy\test\GT\220.png')
    Return:
        ['D:\\', 'PythonProject', 'MB_TaylorFormer', 'DehazeFormer', 'data', 'rshazy', 'test', 'GT', '220.png']
    """
    paths = Path(path)
    path_parts = paths.parts
    return list(path_parts)

def SearchSpecificFilePath(basepath, validExts=None, contains=None):
    """
    遍历目录结构  # loop over the directory structure
    :param basepath: 基础路径，从该目录开始递归地遍历其所有子目录
    :param validExts: 指定允许的文件扩展名。如果传入了这个参数，
                     函数只会返回具有这些扩展名的文件路径；如果为 None，则会返回所有文件的路径。
    :param contains: 指定文件名中必须包含的特定字符串。如果传入了这个参数，函数只会返回文件名中
                    包含该字符串的文件路径；如果为 None，则不进行这一过滤。
    """
    for (rootDir, dirNames, filenames) in os.walk(basepath):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue
            ext = filename[filename.rfind("."):].lower()

            if validExts is None or ext.endswith(validExts):
                imagePath = os.path.join(rootDir, filename)
                yield imagePath

def getSpecificImages(basepath, contains=None):
    """
    调用的 SearchSpecificFilePath ,可返回含有特殊字符的文件路径
    :param basepath: 基础路径，从该目录开始递归地遍历其所有子目录
    :param contains: 指定文件名中必须包含的特定字符串。如果传入了这个参数，函数只会返回文件名中
                    包含该字符串的文件路径；如果为 None，则不进行这一过滤。
    """
    return list(SearchSpecificFilePath(basepath,
                                       validExts=('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'),
                                       contains=contains))

if __name__ == "__main__":
    path = r"E:\PythonProject\clip_pytorch\utils\image_test"
    print(get_image_path(path))
    print(DeepSearchFilePath(path))
    print(SearchFilePath(path))
    print(SearchFileName(path))
    print(split_path(path))
    print(getSpecificImages(path, contains='jpg'))