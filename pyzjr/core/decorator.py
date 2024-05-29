import time
from functools import wraps

def timing(decimal=5):
    """
    计时器装饰器，用于测量函数执行的时间。
    Args:
        decimal: 时间保留的小数位数。
    Returns:
        被装饰后的函数，会在执行前后记录时间，并打印执行时间。
    """
    def decorator(function):
        @wraps(function)
        def timingwrap(*args, **kwargs):
            print(function.__name__, flush=True)
            start = time.perf_counter()
            res = function(*args, **kwargs)
            end = time.perf_counter()
            execution_time = end - start
            format_string = "{:.{}f}".format(execution_time, decimal)
            print(function.__name__, "delta time (s) =", format_string, flush=True)
            return res
        return timingwrap
    return decorator

if __name__=="__main__":
    @timing(decimal=5)
    def test_function():
        time.sleep(2.5)
    test_function()
