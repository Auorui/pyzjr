import yaml
import json
from pyzjr.core.general import is_not_none

def yamlread(path, name=None):
    with open(path) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    value = data[name] if is_not_none(name) and name in data else data

    return value

def jsonread(path, name=None):
    with open(path, 'r') as file:
        data = json.load(file)
    value = data[name] if is_not_none(name) and name in data else data

    return value
