import os
from pyzjr.data.utils import natsorted

def get_file_list(file_dir):
    """获取指定文件夹下的文件列表，并按自然排序方式排序."""
    files = natsorted(os.listdir(file_dir))
    files_num = len(files)
    return files, files_num

def generate_txt(target_path, onlybasename=False, txt_path=r'./output.txt'):
    """将指定文件夹下的文件路径（或文件名）写入到指定的文本文件中"""
    f = open(txt_path, "w")
    files, files_num = get_file_list(target_path)
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

def read_file_from_txt(txt_path):
    """从读取的文件中获取列表"""
    files = []
    for line in open(txt_path, "r"):
        files.append(line.strip())
    return files

def write_to_txt(file_path, content, encoding='utf-8'):
    """将指定的内容写入到指定的文本文件中"""
    with open(file_path, 'w', encoding=encoding) as file:
        file.write(content)

if __name__ == "__main__":
    file_dir = r"D:\PythonProject\pyzjrPyPi\pyzjr\data\file"
    generate_txt(file_dir, onlybasename=True)
    print(read_file_from_txt("./output.txt"))