import os
from pyzjr.data.utils.listfun import natsorted
from pyzjr.utils.check import is_str, is_list_or_tuple

def generate_txt(target_path, onlybasename=False, txt_path=r'./output.txt'):
    """将指定文件夹下的文件路径（或文件名）写入到指定的文本文件中"""
    f = open(txt_path, "w")
    files = natsorted(os.listdir(target_path))
    files_num = len(files)
    index_count = 0
    count = 0
    for file in files:
        index_count = index_count + 1
        path = os.path.splitext(file)[0] if onlybasename else os.path.join(target_path, str(file))
        if count == files_num - 1:
            f.write(path)
            break
        if index_count >= 0:
            f.write(path + "\n")
            count = count + 1
    f.close()

def read_txt(txt_path):
    """从读取的文件中获取列表"""
    path_list = []
    if isinstance(txt_path, str):  # Check if txt_path is a string
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            path_list.extend(line.strip() for line in lines)
    return path_list

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

if __name__ == "__main__":
    file_dir = r"D:\PythonProject\pyzjrPyPi\pyzjr\data"
    generate_txt(file_dir, onlybasename=True)
    print(read_txt("./output.txt"))