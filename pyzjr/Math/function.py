"""
time: 2024-02-19
处于大三的寒假, 正在看张宇的考研数学, 空余时间复现的泰勒展开式近似重要函数, 以及书上涉及的一些重要函数
参考书籍《张宇考研数学基础30讲》 P-27 (2025版)
"""
import math
import pyzjr.Math.constant as const
from pyzjr.Math.arithmetic import odd_factorial, even_factorial

__all__ = ["cos", "sin", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "angle_to_2pi",
           "to_degree", "to_radians", "exp", "log_e", "sinh", "arsinh", "cosh", "arcosh",
           "sigmoid", "tanh", "artanh", "relu"]

def cos(x):
    """弧度制 cos"""
    return math.cos(x)

def sin(x):
    """弧度制 sin"""
    return math.sin(x)

def tan(x):
    """弧度制 tan"""
    return math.tan(x)

def cot(x):
    tan_value = tan(x)
    if tan_value == 0:
        raise ValueError("cot(x) is undefined when tan(x) is zero.")
    return 1 / tan_value

def sec(x):
    """弧度制 sec"""
    cos_value = cos(x)
    if cos_value == 0:
        raise ValueError("sec(x) is undefined when cos(x) is zero.")
    return 1 / cos_value

def csc(x):
    """弧度制 csc"""
    sin_value = sin(x)
    if sin_value == 0:
        raise ValueError("csc(x) is undefined when sin(x) is zero.")
    return 1 / sin_value

def arccos(x):
    """计算反正弦值"""
    return math.acos(x)

def arcsin(x):
    """计算反正弦值"""
    return math.asin(x)

def arctan(x):
    """计算反正弦值"""
    return math.atan(x)

def angle_to_2pi(angle):
    """将角度映射到0到2π"""
    two_pi = 2 * const.pi
    return angle % two_pi

def to_degree(radians_value):
    """弧度转度数"""
    return math.degrees(radians_value)

def to_radians(angle_degrees):
    """度数转弧度"""
    return math.radians(angle_degrees)

def exp(x):
    """计算以e为底的指数函数值"""
    return (const.e) ** x

def log_e(x):
    """计算以e为底的对数函数值"""
    return math.log(x)

def sinh(x):
    """双曲正弦函数"""
    return (exp(x) - exp(-x)) / 2

def arsinh(x):
    """反双曲正弦函数"""
    return log_e(x + math.sqrt(x**2 + 1))

def cosh(x):
    """双曲余弦函数"""
    return (exp(x) + exp(-x)) / 2

def arcosh(x):
    """反双曲余弦函数"""
    return log_e(x + math.sqrt(x**2 - 1))

def tanh(x):
    """双曲正切函数"""
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

def artanh(x):
    """反双曲正切函数"""
    return 0.5 * log_e((1+x) / (1-x))

def sigmoid(x):
    """Sigmoid 函数"""
    return 1 / (1 + exp(-x))

def relu(x):
    """ReLU 激活函数"""
    return max(0, x)