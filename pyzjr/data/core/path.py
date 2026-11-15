"""
Copyright (c) 2024, Auorui.
All rights reserved.

Locating the file path
"""
import re
import os
from pathlib import Path


IMG_EXTENSIONS = ('.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm')


def natsorted(a):
    """
    Sort a list in natural order.

    Example:
        >>> a = ['num9', 'num5', 'num2']
        >>> natsorted(a)
        ['num2', 'num5', 'num9']
    """
    def _natural_key(st):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', st)]

    return sorted(a, key=_natural_key)


def list_dirs(dir_path: str, return_abs_path: bool = False):
    """
    List all directories at a given root.

    Args:
        dir_path (str): Path to directory whose folders need to be listed.
        return_abs_path (bool, optional): If True, returns the full path of each directory.
                                    Otherwise, only returns the directory names. Defaults to False.

    Returns:
        list: Naturally sorted list of directories.
    """
    root = os.path.expanduser(dir_path)
    directories = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]

    if return_abs_path:
        directories = [os.path.join(root, d) for d in directories]

    return natsorted(directories)


def list_files(dir_path: str, suffix=None, return_abs_path: bool = False):
    """
    List all files at a given root, optionally filtering by suffix.

    Args:
        dir_path (str): Path to directory whose files need to be listed.
        suffix (str or tuple, optional): Suffix of the files to match (e.g., '.png' or ('.jpg', '.png')).
                                         If None, all files will be listed. Defaults to None.
        return_abs_path (bool, optional): If True, returns the full path of each file.
                                    Otherwise, only returns the file names. Defaults to False.

    Returns:
        list: Naturally sorted list of files.
    """
    root = os.path.expanduser(dir_path)
    files = [p for p in os.listdir(root) if os.path.isfile(os.path.join(root, p))]

    if suffix is not None:
        if isinstance(suffix, list):
            suffix = tuple(suffix)
        files = [p for p in files if p.endswith(suffix)]

    if return_abs_path:
        files = [os.path.join(root, d) for d in files]

    return natsorted(files)


def find_files(dir_path: str, suffix=None, contains: str = None, return_abs_path: bool = True):
    """
    Recursively search for files in a directory and its subdirectories.

    Args:
        dir_path (str): Root directory to search from.
        suffix (str or tuple/list, optional): Allowed file extensions (e.g., '.png').
        contains (str, optional): A substring that must be present in the filename (case-insensitive).
        return_abs_path (bool): If True, returns the full absolute path of each matching file.
                                If False, returns only the filenames.

    Returns:
        list: Naturally sorted list of matching file paths/names.
    """
    root = os.path.expanduser(dir_path)
    results = []

    for r, _, files in os.walk(root):
        for f in files:
            if suffix is not None:
                if isinstance(suffix, list):
                    suffix = tuple(suffix)
                suffixes = tuple(s.lower() for s in suffix) if isinstance(suffix, tuple) else suffix.lower()
                if not f.lower().endswith(suffixes):
                    continue

            if contains is not None and contains.lower() not in f.lower():
                continue

            results.append(os.path.join(r, f) if return_abs_path else f)

    return natsorted(results)


def split_path(path, resolve=False):
    """
    Split a file path into its components using pathlib.

    This function works perfectly across different operating systems
    (Windows, Linux, macOS) by handling slashes and drive letters automatically.

    Args:
        path (str): File path to split.
        resolve (bool): If True, converts the path to an absolute path before splitting.
                        This resolves '..' and '.' relative paths. Defaults to True.

    Returns:
        list: Path components as separate strings.

    Example:
        >>> split_path('D:/PythonProject/data/220.png')
        ['D:\\', 'PythonProject', 'data', '220.png']
        >>> split_path('./data/test.png', resolve=True)
        ['/', 'home', 'user', 'project', 'data', 'test.png']  # (On Windows, it is ['D:\\', ...])
    """
    p = Path(path)
    if resolve:
        p = p.resolve()
    return list(p.parts)


def get_image_path(dir_path: str, return_abs_path: bool = False):
    """
    Retrieve paths of all common image files in a given directory (non-recursive).

    Args:
        dir_path (str): Directory to search.
        return_abs_path (bool): If True, returns the full path. Defaults to True.

    Returns:
        list: Naturally sorted list of image file paths.
    """
    return list_files(dir_path, suffix=IMG_EXTENSIONS, return_abs_path=return_abs_path)


if __name__=="__main__":
    paths = r'E:\PythonProject\pyzjrPyPi'
    dirs = list_dirs(paths)
    print(dirs)
    files = list_files(paths, suffix=('.py', '.md'))
    print(files)
    image_path = get_image_path(r'E:\PythonProject\pyzjrPyPi\resources\images', return_abs_path=True)
    print(image_path)
