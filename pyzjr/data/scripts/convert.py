import os

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

if __name__=="__main__":
    file_path = r"D:\PythonProject\MB_TaylorFormer\DehazeFormer\data\rshazy\test\GT\220.png"
    new_path = convert_suffix(file_path, 'jpg')
    print(new_path)




