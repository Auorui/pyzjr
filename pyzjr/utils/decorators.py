"""
Copyright (c) 2023, Auorui.
All rights reserved.
"""
import time
import numpy as np
from functools import wraps


def _sync_wrapper(func):
    """
    Decorator: Automatically compatible with single-image and multi-image lists.
    - If it is a single image, wrap it in [img] before passing it in, and extract img upon return.
    - If it is multiple images, pass them directly as a list.
    """
    def wrapper(images, *args, **kwargs):
        if isinstance(images, np.ndarray):
            result = func([images], *args, **kwargs)
            return result[0]
        elif isinstance(images, (list, tuple)):
            return func(list(images), *args, **kwargs)
        else:
            raise TypeError("Input must be a NumPy array or a list/tuple of NumPy arrays.")
    return wrapper


def timing(decimal=5):
    """A timer decorator used to measure the execution time of a function."""
    def decorator(function):
        @wraps(function)
        def timingwrap(*args, **kwargs):
            start = time.perf_counter()
            res = function(*args, **kwargs)
            end = time.perf_counter()
            execution_time = end - start
            format_string = "{:.{}f}".format(execution_time, decimal)
            print(f"[{function.__name__}] Execution time: {format_string} s", flush=True)
            return res
        return timingwrap
    return decorator


if __name__=="__main__":
    @timing(decimal=5)
    def test_function():
        time.sleep(2.5)
    test_function()