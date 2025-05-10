"""
Copyright (c) 2023, Auorui.
All rights reserved.
"""
import os
import cv2
import sys

class VideoCap():
    """
    Customized Python video reading class
    Examples:
    ```
        Vcap = VideoCap(mode=0)
        while True:
            img = Vcap.read()
            Vcap.show("ss", img)
    ```
    """
    def __init__(self, mode=0, width=640, height=480, light=150):
        self.cap = cv2.VideoCapture(mode)
        self.cap.set(3, width)
        self.cap.set(4, height)
        self.cap.set(10, light)
        self.start_number = 0

    def read(self, flip=None):
        """
        :param flip: -1: Horizontal and vertical directions,
                      0: along the y-axis, vertical,
                      1: along the x-axis, horizontal
        """
        _, img = self.cap.read()
        if flip is not None:
            assert flip in [-1, 0, 1], f"VideoCap: The 'flip' parameter must be -1, 0, or 1."
            img = cv2.flip(img, flip)
        return img

    def free(self):
        """
        Release camera
        """
        self.cap.release()
        cv2.destroyAllWindows()

    def show(self, winname, src, base_name: str = './result.png', end_k=27,
             save_k=ord('s'), delay_t=1, extend_num=3):
        """
        Window display. Press 's' to save, 'Esc' to end
        """
        image_path, ext = os.path.splitext(base_name)
        os.makedirs(os.path.dirname(base_name), exist_ok=True)
        if src is not None:
            cv2.imshow(winname, src)
            k = cv2.waitKey(delay_t) & 0xFF
            if k == end_k:
                self.free()
                sys.exit(0)
            elif k == save_k:
                self.start_number += 1
                file_number = str(self.start_number).zfill(extend_num)
                file_path = f"{image_path}_{file_number}{ext}"
                print(f"{self.start_number}  Image saved to {file_path}")
                cv2.imwrite(file_path, src)



if __name__=="__main__":
    Vcap = VideoCap(mode=0)
    while True:
        img = Vcap.read()
        Vcap.show("ss", img)