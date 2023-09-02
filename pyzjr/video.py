"""
- author:Auorui(夏天是冰红茶)
- creation time:2022.12.21
"""
import time
import cv2
import imageio
import numpy as np
import pyzjr.Z as Z

class VideoCap:
    """
    Customized Python video reading class
    """
    def CapInit(self,mode=0,w=640,h=480,l=150):
        self.cap = cv2.VideoCapture(mode)
        self.cap.set(3, w)
        self.cap.set(4, h)
        self.cap.set(10, l)
    def read(self, flip=None):
        """
        :param flip: -1: Horizontal and vertical directions,
                      0: along the y-axis, vertical,
                      1: along the x-axis, horizontal
        """
        _, img = self.cap.read()
        if flip is not None:
            if flip in [-1, 0, 1]:
                img = cv2.flip(img, flip)
            else:
                raise ValueError("[pyzjr]:The flip parameter should be -1, 0, or 1")
        return img
    def free(self):
        """
        Release camera
        """
        self.cap.release()

class FPS:
    def __init__(self):
        self.pTime = time.time()

    def update(self, img=None, pos=(20, 50), color=(255, 0, 0), scale=3, thickness=3):
        """
        更新帧速率
        :param img: 显示的图像,如果只需要fps值,则可以留空
        :param pos: 图像上FPS上的位置
        :param color: 显示的FPS值的颜色
        :param scale: 显示的FPS值的比例
        :param thickness: 显示的FPS值的厚度
        :return:
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
        """在创建时就start就被调用了"""
        self.times = []
        self.start()
    def start(self):
        self.gap = time.time()
    def stop(self):
        """从开始计时到调用 stop 方法的时间间隔"""
        self.times.append(time.time() - self.gap)
        return self.times[-1]
    def avg(self):
        """平均耗时"""
        return sum(self.times) / len(self.times)
    def total(self):
        """总耗时"""
        return sum(self.times)
    def cumsum(self):
        """前n次运行的时间累积和
        [2.3, 4.1, 7.1, 9.6] 秒。这表示前1次运行的时间为2.3秒,
        前2次运行的时间为2.3秒+1.8秒=4.1秒,前3次运行的时间为2.3秒+1.8秒+3.0秒=7.1秒,
        前4次运行的时间为2.3秒+1.8秒+3.0秒+2.5秒=9.6秒
        """
        return np.array(self.times).cumsum().tolist()


def Mp4toGif(mp4, name='result.gif', fps=10, start=None, end=None):

    cap = cv2.VideoCapture(mp4)
    all_images = []
    frame_count = 0

    while True:
        ret, img = cap.read()
        if ret is False:
            break
        if start is not None and frame_count < start:
            frame_count += 1
            continue
        if end is not None and frame_count >= end:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        all_images.append(img)
        frame_count += 1

    duration = int(1000 / fps)  # 将帧率转换为每帧之间的延迟时间（毫秒）
    gif = imageio.mimsave(name, all_images, duration=duration)
    print("转换完成！")

def main():
    fpsReader = FPS()
    while True:
        time.sleep(0.025)
        fps = fpsReader.update()
        print(fps)


def mainWebcam():
    fpsReader = FPS()
    Vcap = VideoCap()
    Vcap.CapInit(mode=0, w=480, h=480)
    while True:
        img = Vcap.read()
        fps, img = fpsReader.update(img)
        cv2.imshow("Image", img)
        k = cv2.waitKey(1)
        if k == Z.Esc:
            break


if __name__ == "__main__":
    # main()
    mainWebcam()
