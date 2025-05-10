"""
time: 2024-02-19

套用 math, 实现一些常用数学逻辑运算
"""
import math
import pyzjr.Z as Z
import numpy as np
from decimal import Decimal

def EuclidDistance(point1, point2):
    """欧式距离, 即两点间距离公式"""
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def ChessboardDistance(point1, point2):
    """棋盘距离"""
    return abs(point2[0] - point1[0]) + abs(point2[1] - point1[1])

def CityblockDistance(point1, point2):
    """Cityblock距离"""
    return max(abs(point2[0] - point1[0]), abs(point2[1] - point1[1]))

def CenterPointDistance(point1, point2):
    """中心点计算公式"""
    return (int((point1[0] + point2[0]) / 2), int((point1[1] + point2[1]) / 2))

def normal(x, mu, sigma):
    """正态分布（高斯分布）概率密度函数"""
    p = 1 / np.sqrt(Z.double_pi * sigma**2)
    return p * exp(-0.5 / sigma**2 * (x - mu)**2)

def gaussian2d(x, y, sigma):
    # 计算二维高斯函数的数值
    normalization = 1 / ((Z.double_pi ** 0.5) * sigma)
    exponent = - ((x**2 + y**2) / (2 * sigma**2))
    result = normalization * exp(exponent)
    return result

def retain(x, t=2):
    """精准保留小数,默认2位"""
    return round(Decimal(x), t)

def round_up(x):
    """向上取整"""
    return math.ceil(x)

def round_down(x):
    """向下取整"""
    return math.floor(x)

def factorial(n):
    # 阶乘
    return math.factorial(n)

def pow(x, n):
    """幂函数计算"""
    return x ** n

def sqrt(x):
    """二次方根"""
    return pow(x, 2)

def rsqrt(x):
    """二次方根倒数"""
    return 1 / sqrt(x)

def cos(x):
    return math.cos(x)

def sin(x):
    return math.sin(x)

def tan(x):
    return math.tan(x)

def cot(x):
    tan_value = tan(x)
    if tan_value == 0:
        raise ValueError("cot(x) is undefined when tan(x) is zero.")
    return 1 / tan_value

def sec(x):
    cos_value = cos(x)
    if cos_value == 0:
        raise ValueError("sec(x) is undefined when cos(x) is zero.")
    return 1 / cos_value

def csc(x):
    sin_value = sin(x)
    if sin_value == 0:
        raise ValueError("csc(x) is undefined when sin(x) is zero.")
    return 1 / sin_value

def arccos(x):
    return math.acos(x)

def arcsin(x):
    return math.asin(x)

def arctan(x):
    return math.atan(x)

def angle_to_2pi(angle):
    return angle % Z.double_pi

def to_degree(radians_value):
    return math.degrees(radians_value)

def to_radians(angle_degrees):
    return math.radians(angle_degrees)

def exp(x):
    return Z.e ** x

def log_e(x):
    return math.log(x)

def sinh(x):
    return (exp(x) - exp(-x)) / 2

def arsinh(x):
    return log_e(x + math.sqrt(x**2 + 1))

def cosh(x):
    return (exp(x) + exp(-x)) / 2

def arcosh(x):
    return log_e(x + math.sqrt(x**2 - 1))

def tanh(x):
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

def artanh(x):
    return 0.5 * log_e((1+x) / (1-x))

def sigmoid(x):
    return 1 / (1 + exp(-x))

def relu(x):
    return max(0, x)