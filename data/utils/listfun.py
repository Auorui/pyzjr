import re
import os


def list_alphabet(capital=False):
    """字符标签"""
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    uppercase_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    return uppercase_alphabet if capital else alphabet

def natural_key(st):
    """
    将字符串拆分成字母和数字块
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', st)]

def natsorted(a):
    """
    手写自然排序
    >>> a = ['num9', 'num5', 'num2']
    >>> sorted_a = natsorted(a)
    ['num2', 'num5', 'num9']
    """
    return sorted(a, key=natural_key)

def list_dirs(root: str, prefix: bool = False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]
    return natsorted(directories)

def list_files(root, suffix=None, prefix=False):
    """List all files at a given root, optionally filtering by suffix.

    Args:
        root (str): Path to directory whose files need to be listed.
        suffix (str or tuple or None, optional): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            If None, all files will be listed. It uses the Python "str.endswith" method and is passed directly.
        prefix (bool, optional): If True, prepends the path to each result, otherwise
            only returns the name of the files found.
    """
    root = os.path.expanduser(root)
    files = [p for p in os.listdir(root) if os.path.isfile(os.path.join(root, p))]
    if suffix is not None:
        files = [p for p in files if p.endswith(suffix)]

    if prefix:
        files = [os.path.join(root, d) for d in files]

    return natsorted(files)


if __name__=="__main__":
    paths = r'E:\PythonProject\pyzjrPyPi'
    dirs = list_dirs(paths)
    print(dirs)
    files = list_files(paths)
    print(files)
