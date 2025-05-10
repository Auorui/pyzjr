"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is used for visualizing useful functions.
"""
import cv2
import time
import numpy as np
from functools import wraps

class FPS:
    """
    Detect video frame rate and refresh the video display
    Examples:
    ```
        fpsReader = FPS()
        Vcap = VideoCap(mode=0)
        while True:
            img = Vcap.read()
            fps, img = fpsReader.update(img)
            Vcap.show("ss", img)
    ```
    """
    def __init__(self):
        self.pTime = time.time()

    def update(self, img=None, pos=(20, 50), color=(255, 0, 0), scale=3, thickness=3):
        """
        Update frame rate
        :param img: The displayed image can be left blank if only the fps value is needed
        :param pos: Position on FPS on image
        :param color: The color of the displayed FPS value
        :param scale: The proportion of displayed FPS values
        :param thickness: The thickness of the displayed FPS value
        """
        cTime = time.time()
        try:
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            if img is None:
                return fps
            else:
                cv2.putText(img, f'FPS: {int(fps)}', pos, cv2.FONT_HERSHEY_PLAIN,
                            scale, color, thickness)
                return fps, img
        except:
            return 0

class Timer:
    def __init__(self):
        """Start is called upon creation"""
        self.times = []
        self.start()

    def start(self):
        """initial time"""
        self.gap = time.time()

    def stop(self):
        """The time interval from the start of timing to calling the stop method"""
        self.times.append(time.time() - self.gap)
        return self.times[-1]

    def avg(self):
        """Average time consumption"""
        return sum(self.times) / len(self.times)

    def total(self):
        """Total time consumption"""
        return sum(self.times)

    def cumsum(self):
        """Accumulated sum of time from the previous n runs"""
        return np.array(self.times).cumsum().tolist()


class Runcodes:
    """
    Comparing the running time of different algorithms.
    Examples:
    ```
        with Runcodes("inference time"):
            output = ...
    ```
    """
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.7f} sec')


def timing(decimal=5):
    """计时器装饰器，用于测量函数执行的时间。"""
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