from itertools import repeat
import collections.abc as container_abcs

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def convert_to_tuple(value):
    if isinstance(value, int):
        return (value, value)
    elif isinstance(value, (list, tuple)):
        if len(value) == 2:
            return value
        else:
            raise ValueError("The input object is of type list or tuple, but the length is not two!")

