import os
import yaml
import json
from pyzjr.data.core.path import natsorted


def read_yaml(path, name=None):
    """
    Read YAML file and return its content.

    Args:
        path (str): Path to the YAML file.
        name (str, optional): If provided, return only the value of this specific key.

    Returns:
        dict/list/any: The parsed YAML content or the specific key's value.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    if name is not None and isinstance(data, dict) and name in data:
        return data[name]
    return data


def read_json(path, name=None):
    """
    Read JSON file and return its content.

    Args:
        path (str): Path to the JSON file.
        name (str, optional): If provided, return only the value of this specific key.

    Returns:
        dict/list/any: The parsed JSON content or the specific key's value.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if name is not None and isinstance(data, dict) and name in data:
        return data[name]
    return data


def read_txt(txt_path: str):
    """
    Read lines from a text file into a list.

    Args:
        txt_path (str): Path to the text file.

    Returns:
        list: A list of lines (stripped of trailing newlines).
    """
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"Text file not found: {txt_path}")

    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


def write_txt(file_path, content, encoding='utf-8'):
    """
    Write the specified content into the designated text file.
    Automatically handles lists, tuples, and strings.

    Args:
        file_path (str): Target file path.
        content (str or list or tuple): Content to write.
        encoding (str): File encoding. Defaults to 'utf-8'.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    if isinstance(content, (list, tuple)):
        with open(file_path, 'w', encoding=encoding) as file:
            for item in content:
                file.write(str(item) + "\n")

    elif isinstance(content, str):
        with open(file_path, 'w', encoding=encoding) as file:
            file.write(content)

    else:
        raise TypeError(f"Content must be str, list, or tuple, got {type(content)}")


def export_file_list(dir_path, file_path='./output.txt', onlybasename=False, suffix=None):
    """
    Scan a directory and save the file paths or filenames to a txt file.

    Args:
        dir_path (str): Directory to scan.
        file_path (str): Target txt file path. Defaults to './output.txt'.
        onlybasename (bool): If True, save only the filename without path and extension.
                             If False, save the full path (with extension). Defaults to False.
        suffix (str or tuple, optional): Filter files by extension (e.g., '.jpg' or ('.png', '.jpg')).
                                         If None, all files will be included.
    """
    ensure_file_dir = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(ensure_file_dir, exist_ok=True)

    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    files = natsorted(os.listdir(dir_path))

    result_list = []
    for file in files:
        file_full_path = os.path.join(dir_path, file)
        if not os.path.isfile(file_full_path):
            continue
        if suffix is not None and not file.lower().endswith(suffix):
            continue
        if onlybasename:
            result_list.append(os.path.splitext(file)[0])
        else:
            result_list.append(file_full_path)
    write_txt(file_path, result_list)


if __name__ == "__main__":
    file_dir = r"D:\PythonProject\pyzjrPyPi\pyzjr\data"
    export_file_list(file_dir, file_path='./output.txt', onlybasename=True, suffix='.py')
    print(read_txt("./output.txt"))