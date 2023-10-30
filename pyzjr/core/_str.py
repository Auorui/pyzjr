import os
import string
import re
from pathlib import Path


__all__=["natural_key","natsorted","Sorting"]

def pathstr(path_str):
    """
    path_str = 'D:/PythonProject/Torchproject/Lasercenterline/line/20231013-LaserLine_txt/test_2/imges/Image_1.jpg'
    D:\
    PythonProject
    Torchproject
    Lasercenterline
    line
    20231013-LaserLine_txt
    test_2
    imges
    Image_1.jpg
    """
    path = Path(path_str)
    path_parts = path.parts
    return list(path_parts)

def natural_key(st):
    """
    将字符串拆分成字母和数字块
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', st)]

def natsorted(a):
    """
    手写自然排序
    >>> a = ['num9', 'num5', 'num2']
    >>> sorted_a = natsorted(a)
    ['num2', 'num5', 'num9']
    """
    return sorted(a, key=natural_key)



class Sorting():
    """排序算法"""
    def insertion_sort(self, arr):
        """直接插入排序"""
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
        """快速排序"""
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
        """冒泡排序"""
        n = len(arr)
        for i in range(n - 1):
            for j in range(n - 1 - i):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
            print(arr)

    def selection_sort(self, arr):
        """直接选择排序"""
        n = len(arr)
        for i in range(n - 1):
            min_index = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_index]:
                    min_index = j
            arr[i], arr[min_index] = arr[min_index], arr[i]
            print(arr)
