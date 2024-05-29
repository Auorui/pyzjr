import numpy as np
from pyzjr.Math.constant import n2pi

def rand(a=0., b=1.):
    """
    生成在指定范围内的随机浮点数,进行缩放和偏移来映射到[a, b)的范围
    a: 下界; b: 上界
    """
    return np.random.rand() * (b - a) + a

def normal(x, mu, sigma):
    """正态分布（高斯分布）概率密度函数"""
    p = 1 / np.sqrt(n2pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

def gaussian2d(x, y, sigma):
    # 计算二维高斯函数的数值
    normalization = 1 / ((n2pi ** 0.5) * sigma)
    exponent = - ((x**2 + y**2) / (2 * sigma**2))
    result = normalization * np.exp(exponent)
    return result

