import cv2 as cv
import mediapipe as mp
from pyzjr.dlearn.strategy import cvtColor
import pyzjr.math as zmath
import pyzjr.Z as Z
from pyzjr.utils import cornerRect

class Hands():
    def __init__(self,NumHands=2,minDetconfid=0.5,minTrackconfid=0.5,StaticMode=False):
        self.NumHands=NumHands
        self.minDetconfid=minDetconfid
        self.minTrackconfid=minTrackconfid
        self.StaticMode=StaticMode
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.StaticMode,
                                        max_num_hands=self.NumHands,
                                        min_detection_confidence=self.minDetconfid,
                                        min_tracking_confidence=self.minTrackconfid)
        self.mpDraw = mp.solutions.drawing_utils
        mark=Z.HandLandmark()
        self.tipIds = [mark.THUMB_TIP, mark.INDEX_FINGER_TIP, mark.MIDDLE_FINGER_TIP, mark.RING_FINGER_TIP, mark.PINKY_TIP]  # 指尖索引

    def findHands(self, img, draw=True, flipType=True, showcor=False):
        """
        在BGR图像中查找手,内部会强制转换为RGB
        :param img: 图像
        :param draw: 在图像上绘制输出的标志
        :return: 带或不带标志的图像
        """
        imgRGB = cvtColor(img)
        h, w, _ = img.shape
        self.results = self.hands.process(imgRGB)
        allHands = []
        if self.results.multi_hand_landmarks:
            for handType,handLms in zip(self.results.multi_handedness,self.results.multi_hand_landmarks):
                myHand={}
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    mylmList.append([px, py])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)


                    if showcor:
                        img=cornerRect(img, bbox=[bbox[0] - 20, bbox[1] - 20, bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20])
                    else:
                        cv.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                     (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                     Z.purple, 2)
                    cv.putText(img,myHand["type"],(bbox[0] - 30, bbox[1] - 30),cv.FONT_HERSHEY_PLAIN,
                                2,Z.purple,2)
        if draw:
            return allHands,img
        else:
            return allHands

    def gesture(self, myHand):
        """
        手势形状,分别考虑左手和右手
        :param myHand: 单个手的信息字典
        :return: 以列表的形式展现手指状态
        """
        myHandType =myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def fingerSpace(self, p1, p2, img=None, color=Z.green, radius = 15, line_thickness = 2):
        """
        计算两个指定点之间的距离，并在图像上绘制相关信息（可选）。
        :param p1: Point1，(x1, y1)
        :param p2: Point2，(x2, y2)
        :param img: 要绘制在上面的图像（可选）
        :param color: 绘制的颜色
        :param radius: 绘制圆的半径
        :param line_thickness:线条厚度
        :return: 如果提供了图像，则返回信息和已绘制的图像
                 如果没有提供图像，则返回信息
        """
        distance, center = zmath.EuclideanDis(p1, p2)
        info = [distance, center]
        if img is not None:
            cv.circle(img, p1, radius, color, cv.FILLED)
            cv.circle(img, p2, radius, color, cv.FILLED)
            cv.line(img, p1, p2, color, line_thickness)
            cv.circle(img, p2, radius, color, cv.FILLED)

            return info, img
        else:
            return info


def mainHands():
    from pyzjr.video import VideoCap
    Vap=VideoCap()
    Vap.CapInit()
    detector = Hands()
    while True:
        img = Vap.read()
        hands, img = detector.findHands(img)
        if hands:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]
            fingers = detector.gesture(hand1)
            if len(hands) == 2:
                hand2 = hands[1]
                lmList2 = hand2["lmList"]
                info, img = detector.fingerSpace(lmList1[8], lmList2[8], img)
        cv.imshow("Image", img)
        if cv.waitKey(1) == Z.Esc:
            break

if __name__=="__main__":
    mainHands()