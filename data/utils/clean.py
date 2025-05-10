import os
import shutil

def clear_directory(directory_path):
    """清空指定文件夹下的所有文件和子文件夹."""
    if not os.path.exists(directory_path):
        print(f"Error: The directory {directory_path} does not exist.")
        return
    for root, dirs, files in os.walk(directory_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                os.rmdir(dir_path)
                print(f"Deleted directory: {dir_path}")
            except Exception as e:
                print(f"Error deleting directory {dir_path}: {e}")

def cleanup_directory(file_path):
    """递归地删除目录及其内容"""
    if os.path.exists(file_path):
        shutil.rmtree(file_path)
        print(f"'{file_path}' has been deleted.")

def rm_makedirs(file_path: str):
    # 如果文件夹存在，则先删除原文件夹在重新创建
    if os.path.exists(file_path):
        shutil.rmtree(file_path)
    os.makedirs(file_path)