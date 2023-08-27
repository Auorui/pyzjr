import numpy as np
from decimal import Decimal

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


class pysort():
    """排序算法"""
    def insertion_sort(self, arr):
        """
        直接插入排序
        """
        n = len(arr)
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
            print(arr)

    def quick_sort(self, arr, low, high):
        """
        快速排序
        """
        if low < high:
            pivot_index = self.partition(arr, low, high)
            print(arr)
            self.quick_sort(arr, low, pivot_index - 1)
            self.quick_sort(arr, pivot_index + 1, high)

    def partition(self, arr, low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    def bubble_sort(self, arr):
        """
        冒泡排序
        """
        n = len(arr)
        for i in range(n - 1):
            for j in range(n - 1 - i):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
            print(arr)

    def selection_sort(self, arr):
        """
        直接选择排序
        """
        n = len(arr)
        for i in range(n - 1):
            min_index = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_index]:
                    min_index = j
            arr[i], arr[min_index] = arr[min_index], arr[i]
            print(arr)
if __name__=="__main__":
    a=[64,45,8,10,70,6]
    Sort=pysort()
    Sort.selection_sort(a)