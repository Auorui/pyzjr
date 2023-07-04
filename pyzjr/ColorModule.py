import cv2
import numpy as np
import logging
from pyzjr.utils import empty


class ColorFinder:
    def __init__(self, trackBar=False, name="Bars"):
        self.trackBar = trackBar
        self.name = name
        if self.trackBar:
            self.initTrackbars()

    def initTrackbars(self):
        """
        :return:初始化轨迹栏
        """
        cv2.namedWindow(self.name)
        cv2.resizeWindow(self.name, 640, 240)
        cv2.createTrackbar("Hue Min", self.name, 0, 179, empty)
        cv2.createTrackbar("Hue Max", self.name, 179, 179, empty)
        cv2.createTrackbar("Sat Min", self.name, 0, 255, empty)
        cv2.createTrackbar("Sat Max", self.name, 255, 255, empty)
        cv2.createTrackbar("Val Min", self.name, 0, 255, empty)
        cv2.createTrackbar("Val Max", self.name, 255, 255, empty)

    def getTrackbarValues(self,showVals=True):
        """
        Gets the trackbar values in runtime
        :return: hsv values from the trackbar window
        """
        hmin = cv2.getTrackbarPos("Hue Min", self.name)
        smin = cv2.getTrackbarPos("Sat Min", self.name)
        vmin = cv2.getTrackbarPos("Val Min", self.name)
        hmax = cv2.getTrackbarPos("Hue Max", self.name)
        smax = cv2.getTrackbarPos("Sat Max", self.name)
        vmax = cv2.getTrackbarPos("Val Max", self.name)

        hsvVals = {"hmin": hmin, "smin": smin, "vmin": vmin,
                   "hmax": hmax, "smax": smax, "vmax": vmax}
        HsvVals=[[hmin, smin, vmin],[hmax, smax, vmax]]
        if showVals:
            print(hsvVals)
            return hsvVals
        else:
            return HsvVals

    def update(self, img, myColor=None):
        """
        :param img: Image in which color needs to be found
        :param hsvVals: List of lower and upper hsv range
        :return: (mask) bw image with white regions where color is detected
                 (imgColor) colored image only showing regions detected
        """
        imgColor = [],
        mask = []

        if self.trackBar:
            myColor = self.getTrackbarValues()

        if isinstance(myColor, str):
            myColor = self.getColorHSV(myColor)

        if myColor is not None:
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([myColor['hmin'], myColor['smin'], myColor['vmin']])
            upper = np.array([myColor['hmax'], myColor['smax'], myColor['vmax']])
            mask = cv2.inRange(imgHSV, lower, upper)
            imgColor = cv2.bitwise_and(img, img, mask=mask)
        return imgColor, mask

    def getColorHSV(self, myColor):

        if myColor == 'red':
            output = {'hmin': 146, 'smin': 141, 'vmin': 77, 'hmax': 179, 'smax': 255, 'vmax': 255}
        elif myColor == 'green':
            output = {'hmin': 44, 'smin': 79, 'vmin': 111, 'hmax': 79, 'smax': 255, 'vmax': 255}
        elif myColor == 'blue':
            output = {'hmin': 103, 'smin': 68, 'vmin': 130, 'hmax': 128, 'smax': 255, 'vmax': 255}
        else:
            output = None
            logging.warning("Color Not Defined")
            logging.warning("Available colors: red, green, blue ")

        return output


def main():
    myColorFinder = ColorFinder(False)
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Custom Orange Color
    hsvVals = {'hmin': 10, 'smin': 55, 'vmin': 215, 'hmax': 42, 'smax': 255, 'vmax': 255}

    while True:
        success, img = cap.read()
        imgRed, _ = myColorFinder.update(img, "red")
        imgGreen, _ = myColorFinder.update(img, "green")
        imgBlue, _ = myColorFinder.update(img, "blue")
        imgOrange, _ = myColorFinder.update(img, hsvVals)

        cv2.imshow("Red", imgRed)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
