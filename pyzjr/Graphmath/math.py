import numpy as np
from decimal import Decimal
import math

def EuclideanDis(pts1, pts2):
    """
    欧式距离与中心点
    :param pts1: 位置(x1,y1)
    :param pts2: 位置(x2,y2)
    :return: (两点间距离, 中心点)
    """
    distance = ((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5
    center = ((pts1[0] + pts2[0]) / 2, (pts1[1] + pts2[1]) / 2)
    return distance, center

def retain(val,t=2):
    """精准保留小数,默认2位"""
    value=Decimal(val)
    return round(value,t)

def normal(x, mu, sigma):
    """正态分布（高斯分布）概率密度函数"""
    p = 1 / np.sqrt(2 * np.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

def Gaussian2D(x, y, sigma):
    # 计算二维高斯函数的数值
    normalization = 1 / (((2 * np.pi)**(0.5)) * sigma)
    exponent = -((x**2 + y**2) / (2 * sigma**2))
    result = normalization * np.exp(exponent)
    return result

def Advancedlist(imglist,mode='T',reverse=False):
    """对列表的高级排列"""
    if mode=="T":
        """平铺
        lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] ——> [1, 2, 3, 4, 5, 6, 7, 8, 9]
        """
        tiled_list = [item for sublist in imglist for item in sublist]
        return tiled_list
    elif mode=="F":
        """首元素
        lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] ——> [1, 4, 7, 2, 5, 8, 3, 6, 9]
        """
        sorted_list_by_first_element = [sublist[i] for i in range(len(imglist[0])) for sublist in imglist]
        return sorted_list_by_first_element
    elif mode=="X":
        """按照x排序,默认是从小到大
        lists = [(1, 4), (3, 8), (5, 2)] ——> [(1, 4), (3, 8), (5, 2)]
        """
        sorted_by_x = sorted(imglist, key=lambda item: item[0], reverse=reverse)
        return sorted_by_x
    elif mode=="Y":
        """按照y排序,默认是从小到大
        lists = [(1, 4), (3, 8), (5, 2)] ——> [(5, 2), (1, 4), (3, 8)]
        """
        sorted_by_y = sorted(imglist, key=lambda item: item[1], reverse=reverse)
        return sorted_by_y

def angle_to_2pi_range(angle):
    """将角度映射到0到2π"""
    two_pi = 2 * math.pi
    return angle % two_pi

def degree_to_radians(angle_degrees):
    """度数转弧度"""
    return math.radians(angle_degrees)

def rand(a=0., b=1.):
    """
    生成在指定范围内的随机浮点数,进行缩放和偏移来映射到[a, b)的范围
    :param a: 下界
    :param b: 上界
    :return: 随机浮点数
    """
    return np.random.rand() * (b - a) + a

class NumericalMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count