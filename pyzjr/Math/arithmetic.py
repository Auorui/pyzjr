import math
from decimal import Decimal

__all__ = ["sqrt", "rsqrt", "odd_factorial", "even_factorial", "factorial", "pow", "retain",
           "round_down", "round_up", "Sum", "Subtract", "Multiply", "Divide",
           "EuclidDistance", "ChessboardDistance", "CityblockDistance", "CenterPoint"]

def sqrt(x):
    """二次方根"""
    return pow(x, 2)

def rsqrt(x):
    """二次方根倒数"""
    return 1 / sqrt(x)

def odd_factorial(n):
    # 奇数的阶乘
    result = 1
    for i in range(1, 2 * n + 1, 2):
        result *= i
    return result

def even_factorial(n):
    # 偶数的阶乘
    result = 1
    for i in range(2, 2 * n + 1, 2):
        result *= i
    return result

def factorial(n):
    # 阶乘
    return math.factorial(n)

def pow(x, n):
    """幂函数计算"""
    return x ** n

def retain(val, t=2):
    """精准保留小数,默认2位"""
    value = Decimal(val)
    return round(value, t)

def round_up(x):
    """向上取整"""
    return math.ceil(x)

def round_down(x):
    """向下取整"""
    return math.floor(x)

def Sum(*args):
    """计算参数的和,支持多个参数或一个一维列表(元组)"""
    if not args:
        raise ValueError("At least one argument is required.")

    if len(args) == 1 and (isinstance(args[0], list) or isinstance(args[0], tuple)):
        return sum(args[0])
    else:
        return sum(args)

def Subtract(*args):
    """计算参数的差,支持多个参数或一个一维列表(元组)"""
    if not args:
        raise ValueError("At least one argument is required.")

    if len(args) == 1 and (isinstance(args[0], list) or isinstance(args[0], tuple)):
        result = args[0][0] - sum(args[0][1:])
    else:
        result = args[0] - sum(args[1:])
    return result

def Multiply(*args):
    """计算参数的乘积,支持多个参数或一个一维列表(元组)"""
    if not args:
        raise ValueError("At least one argument is required.")

    if len(args) == 1 and (isinstance(args[0], list) or isinstance(args[0], tuple)):
        result = 1
        for arg in args[0]:
            result *= arg
    else:
        result = 1
        for arg in args:
            result *= arg
    return result

def Divide(*args):
    """计算参数的商,支持多个参数或一个一维列表(元组)"""
    if len(args) == 1 and (isinstance(args[0], list) or isinstance(args[0], tuple)):
        result = args[0][0]
        for arg in args[0][1:]:
            if arg == 0:
                raise ValueError("Cannot divide by zero.")
            result /= arg
    else:
        result = args[0]
        for arg in args[1:]:
            if arg == 0:
                raise ValueError("Cannot divide by zero.")
            result /= arg
    return result

def EuclidDistance(point1, point2):
    """欧式距离, 即两点间距离公式"""
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def ChessboardDistance(point1, point2):
    """棋盘距离"""
    return abs(point2[0] - point1[0]) + abs(point2[1] - point1[1])

def CityblockDistance(point1, point2):
    """Cityblock距离"""
    return max(abs(point2[0] - point1[0]), abs(point2[1] - point1[1]))

def CenterPoint(point1, point2):
    """中心点计算公式"""
    return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)