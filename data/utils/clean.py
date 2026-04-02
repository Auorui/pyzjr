import os
import shutil

def cleanup_directory(dir_path):
    """递归地删除目录及其内容"""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"'{dir_path}' has been deleted.")

def rm_makedirs(dir_path: str):
    # 如果文件夹存在，则先删除原文件夹再重新创建
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def clear_directory(dir_path):
    """清空指定文件夹下的所有文件和子文件夹."""
    if not os.path.exists(dir_path):
        print(f"Error: The directory {dir_path} does not exist.")
        return
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
        for dir in dirs:
            dir_paths = os.path.join(root, dir)
            try:
                os.rmdir(dir_paths)
                print(f"Deleted directory: {dir_paths}")
            except Exception as e:
                print(f"Error deleting directory {dir_paths}: {e}")


if __name__=="__main__":
    target_path = r'D:\software\Nutbit\NutBit'
    clear_directory(target_path)