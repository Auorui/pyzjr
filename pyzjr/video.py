"""
- author:Auorui(夏天是冰红茶)
- creation time:2022.12.21
"""
import time
import cv2
import imageio

class VideoCap:
    """
    自定义的视频读取类
    """
    def CapInit(self,mode=0,w=640,h=480,l=150):
        self.cap = cv2.VideoCapture(mode)
        self.cap.set(3, w)
        self.cap.set(4, h)
        self.cap.set(10, l)
    def read(self):
        ret, img = self.cap.read()
        return img
    def release(self):
        self.cap.release()

class FPS:
    """
    Helps in finding Frames Per Second and display on an OpenCV Image
    """
    def __init__(self):
        self.pTime = time.time()

    def update(self, img=None, pos=(20, 50), color=(255, 0, 0), scale=3, thickness=3):
        """
        更新帧速率
        :param img: 显示的图像，如果只需要fps值，则可以留空
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
    """
    Without Webcam
    """
    fpsReader = FPS()
    while True:
        time.sleep(0.025)  # add delay to get 40 Frames per second
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
        k=cv2.waitKey(1)
        if k==27:
            break



if __name__ == "__main__":
    # main()
    mainWebcam()
