import os
from pathlib import Path
from pyzjr.data.utils.listfun import natsorted

def get_image_path(dir_path):
    """
    Retrieve image file paths in the specified directory that match common image formats.
    Fixes path separator issues (converts backslashes to forward slashes).
    Reference: https://blog.csdn.net/m0_62919535/article/details/132199978

    Args:
        dir_path (str): Directory path to search for images

    Returns:
        list: Sorted list of image file paths with forward slashes
    """
    imgfile = []
    file_list = os.listdir(dir_path)
    for i in file_list:
        new_path = os.path.join(dir_path, i).replace("\\", "/")
        _, file_ext = os.path.splitext(new_path)
        if file_ext[1:] in ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'):
            imgfile.append(new_path)
    return natsorted(imgfile)

def DeepSearchFilePath(dir_path, file_exts=('.png', '.jpg', '.jpeg')):
    """
    Recursively search for files with specified extension in directory and subdirectories.
    Improved version of SearchFilePath with deep search capability.

    Args:
        dir_path (str): Root directory to search from
        file_exts (str): File extension to search for (including the dot)

    Returns:
        list: Naturally sorted list of matching file paths
    """
    search_file_path = []
    for root, dirs, files in os.walk(dir_path):
        for filespath in files:
            if str(filespath).endswith(file_exts):
                search_file_path.append(os.path.join(root, filespath))
    return natsorted(search_file_path)

def SearchFilePath(dir_path, file_exts=('.png', '.jpg', '.jpeg')):
    """
    Search for files with specified extensions in the target directory (non-recursive).

    Args:
        dir_path (str): Directory to search
        file_exts (tuple): File extensions to match (including dots)

    Returns:
        list: Naturally sorted list of matching file paths
    """
    search_file_path = []
    files = os.listdir(dir_path)
    for filespath in files:
        if str(filespath).endswith(file_exts):
            search_file_path.append(os.path.join(dir_path, filespath))
    return natsorted(search_file_path)

def SearchFileName(dir_path, file_exts=('.png', '.jpg', '.jpeg')):
    """
    Get filenames with specified extensions in the target directory (non-recursive).

    Args:
        dir_path (str): Directory to search
        file_exts (tuple): File extensions to match (including dots)

    Returns:
        list: Naturally sorted list of matching filenames
    """
    all_files = os.listdir(dir_path)
    image_files = [file for file in all_files if file.lower().endswith(file_exts)]
    return natsorted(image_files)

def split_path(path):
    """
    Split a file path into its components using pathlib.

    Example:
        path_list = split_path('D:/PythonProject/MB_TaylorFormer/DehazeFormer/data/rshazy/test/GT/220.png')
        Returns: ['D:\\', 'PythonProject', 'MB_TaylorFormer', 'DehazeFormer', 'data', 'rshazy', 'test', 'GT', '220.png']

    Args:
        path (str): File path to split

    Returns:
        list: Path components as separate strings
    """
    paths = Path(path)
    path_parts = paths.parts
    return list(path_parts)

def SearchSpecificFilePath(dir_path, validExts=None, contains=None):
    """
    Generator that recursively searches directory structure for specific files.

    Args:
        dir_path (str): Root directory to start search from
        valid_exts (tuple): Allowed file extensions (None for all files)
        contains (str): String that must be present in filename (None to skip this filter)

    Yields:
        str: Full path of each matching file
    """
    for (rootDir, dirNames, filenames) in os.walk(dir_path):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue
            ext = filename[filename.rfind("."):].lower()

            if validExts is None or ext.endswith(validExts):
                imagePath = os.path.join(rootDir, filename)
                yield imagePath

def getSpecificImages(dir_path, contains=None):
    """
    Wrapper for search_specific_file_path that returns image files with optional name filtering.

    Args:
        dir_path (str): Root directory to start search from
        contains (str): String that must be present in filename (None to skip this filter)

    Returns:
        list: All matching image file paths
    """
    return list(SearchSpecificFilePath(dir_path,
                                       validExts=('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'),
                                       contains=contains))

if __name__ == "__main__":
    path = r"/pyzjr/data/tryout\images"
    print(get_image_path(path))
    print(DeepSearchFilePath(path))
    print(SearchFilePath(path, file_exts='.jpg'))
    print(SearchFileName(path, file_exts='.jpg'))
    print(split_path(path))
    print(getSpecificImages(path, contains='jpg'))