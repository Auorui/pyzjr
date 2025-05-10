from datetime import datetime
from pathlib import Path
import yaml
import json

def read_yaml(path, name=None):
    with open(path) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    value = data[name] if name is not None and name in data else data

    return value

def read_json(path, name=None):
    with open(path, 'r') as file:
        data = json.load(file)
    value = data[name] if name is not None and name in data else data

    return value

def file_age(path, detail=False):
    """Returns the number of days since the last file update."""
    dt = (datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime))  # delta
    se = 0
    if detail:
        se = dt.seconds / 86400
    return f"{dt.days + se} days"

def file_date(path=__file__):
    """Return readable file modification date,i.e:'2021-3-26'."""
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'

def file_size(path):
    """Return file/dir size (MB)."""
    if isinstance(path, (str, Path)):
        mb = 1 << 20  # bytes -> MiB (1024 ** 2)
        path = Path(path)
        if path.is_file():
            size_mb = path.stat().st_size / mb
            return f"{size_mb:.5f} MB"
        elif path.is_dir():
            total_size = sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())
            size_mb = total_size / mb
            return f"{size_mb:.5f} MB"
    return "0.0 MB"



if __name__ == "__main__":
    # 测试 file_age
    current_file_path = Path(__file__)
    age = file_age(current_file_path)
    print(f"The file {current_file_path} is {age} old.")
    detail_age = file_age(current_file_path, detail=True)
    print(f"The file {current_file_path} is {detail_age} old (with detail).")

    # 测试 file_date
    date = file_date(current_file_path)
    print(f"The file {current_file_path} was modified on {date}.")

    # 测试 file_size (文件)
    file_to_check = Path(__file__)
    size = file_size(file_to_check)
    print(f"The file {file_to_check} is {size} large.")

    # 测试 file_size (目录)
    dir_to_check = Path('.')
    dir_size = file_size(dir_to_check)
    print(f"The directory {dir_to_check} is {dir_size} large.")
