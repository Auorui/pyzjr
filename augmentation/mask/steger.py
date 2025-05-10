"""
Copyright (c) 2023, Auorui.
All rights reserved.

This module is used to extract the laser centerline using the steger algorithm
"""
import cv2
import numpy as np

def _derivation_with_Filter(Gaussimg):
    dx = cv2.filter2D(Gaussimg, -1, kernel=np.array([[1], [0], [-1]]))
    dy = cv2.filter2D(Gaussimg, -1, kernel=np.array([[1, 0, -1]]))
    dxx = cv2.filter2D(Gaussimg, -1, kernel=np.array([[1], [-2], [1]]))
    dyy = cv2.filter2D(Gaussimg, -1, kernel=np.array([[1, -2, 1]]))
    dxy = cv2.filter2D(Gaussimg, -1, kernel=np.array([[1, -1], [-1, 1]]))

    return dx, dy, dxx, dyy, dxy

def _derivation_with_Scharr(Gaussimg):
    dx = cv2.Scharr(Gaussimg, cv2.CV_32F, 1, 0)
    dy = cv2.Scharr(Gaussimg, cv2.CV_32F, 0, 1)
    dxx = cv2.Scharr(dx, cv2.CV_32F, 1, 0)
    dxy = cv2.Scharr(dx, cv2.CV_32F, 0, 1)
    dyy = cv2.Scharr(dy, cv2.CV_32F, 0, 1)

    return dx, dy, dxx, dyy, dxy

def _derivation_with_Sobel(Gaussimg):
    dx = cv2.Sobel(Gaussimg, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(Gaussimg, cv2.CV_32F, 0, 1, ksize=3)
    dxx = cv2.Sobel(dx, cv2.CV_32F, 1, 0, ksize=3)
    dxy = cv2.Sobel(dx, cv2.CV_32F, 0, 1, ksize=3)
    dyy = cv2.Sobel(dy, cv2.CV_32F, 0, 1, ksize=3)

    return dx, dy, dxx, dyy, dxy

def Magnitudefield(dxx, dyy):
    """计算幅度和相位"""
    dxx = dxx.astype(float)
    dyy = dyy.astype(float)
    mag = cv2.magnitude(dxx, dyy)
    phase = mag*180./np.pi
    return mag, phase

def derivation(gray_img, sigmaX, sigmaY, method="Scharr", nonmax=False):
    """
    计算图像的一阶导数 dx 和 dy，以及二阶导数 dxx、dyy 和 dxy
    :param gray_img: 灰度图
    :param sigmaX: 在水平方向的高斯核标准差,用于激光线提取建议取1-2
    :param sigmaY: 在垂直方向上的高斯核标准差,用于激光线提取建议取1-2
    :param method:"Scharr"  or  "Filter"  or  "Sobel"
                  选择什么方式获取dx, dy, dxx, dyy, dxy,提供了卷积与Scharr滤波器两种方式计算,
                  Scharr滤波器通常会产生更平滑和准确的结果,所以这里默认使用"Scharr"方法,虽然
                  "Sobel"运行比另外两种要慢,但在使用的时候,建议自己试试
    :return: dx, dy, dxx, dyy, dxy
    """
    Gaussimg = cv2.GaussianBlur(gray_img, ksize=(0, 0), sigmaX=sigmaX, sigmaY=sigmaY)
    if method == "Scharr":
        dx, dy, dxx, dyy, dxy = _derivation_with_Scharr(Gaussimg)
    elif method == "Filter":
        dx, dy, dxx, dyy, dxy = _derivation_with_Filter(Gaussimg)
    elif method == "Sobel":
        dx, dy, dxx, dyy, dxy =_derivation_with_Sobel(Gaussimg)
    if nonmax:
        normal, phase = Magnitudefield(dxx, dyy)
        dxy = nonMaxSuppression(normal, phase)
    return dx, dy, dxx, dyy, dxy

def nonMaxSuppression(det, phase):
    """非最大值抑制"""
    gmax = np.zeros(det.shape)
    # thin-out evry edge for angle = [0, 45, 90, 135]
    for i in range(gmax.shape[0]):
        for j in range(gmax.shape[1]):
            if phase[i][j] < 0:
                phase[i][j] += 360
            if ((j+1) < gmax.shape[1]) and ((j-1) >= 0) and ((i+1) < gmax.shape[0]) and ((i-1) >= 0):
                # 0 degrees
                if (phase[i][j] >= 337.5 or phase[i][j] < 22.5) or (phase[i][j] >= 157.5 and phase[i][j] < 202.5):
                    if det[i][j] >= det[i][j + 1] and det[i][j] >= det[i][j - 1]:
                        gmax[i][j] = det[i][j]
                # 45 degrees
                if (phase[i][j] >= 22.5 and phase[i][j] < 67.5) or (phase[i][j] >= 202.5 and phase[i][j] < 247.5):
                    if det[i][j] >= det[i - 1][j + 1] and det[i][j] >= det[i + 1][j - 1]:
                        gmax[i][j] = det[i][j]
                # 90 degrees
                if (phase[i][j] >= 67.5 and phase[i][j] < 112.5) or (phase[i][j] >= 247.5 and phase[i][j] < 292.5):
                    if det[i][j] >= det[i - 1][j] and det[i][j] >= det[i + 1][j]:
                        gmax[i][j] = det[i][j]
                # 135 degrees
                if (phase[i][j] >= 112.5 and phase[i][j] < 157.5) or (phase[i][j] >= 292.5 and phase[i][j] < 337.5):
                    if det[i][j] >= det[i - 1][j - 1] and det[i][j] >= det[i + 1][j + 1]:
                        gmax[i][j] = det[i][j]
    return gmax


class Steger():
    def __init__(self, image, method="Scharr", usenonmax=True):
        self.image = image.copy()
        self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.row, self.col = self.gray_image.shape[:2]
        self.newimage = np.zeros((self.row, self.col), np.uint8)
        self.method = method
        self.usenonmax = usenonmax

    def centerline(self, sigmaX, sigmaY, color=(0, 0, 255)):
        dx, dy, dxx, dyy, dxy = derivation(self.gray_image, sigmaX, sigmaY, method=self.method, nonmax=self.usenonmax)
        point, direction, value = self.HessianMatrix(dx, dy, dxx, dyy, dxy)
        for point in point:
            self.newimage[point[1], point[0]] = 255
            self.image[point[1], point[0], :] = color

        return self.image, self.newimage

    def HessianMatrix(self, dx, dy, dxx, dyy, dxy, threshold=0.5):
        """
        HessianMatrix = [dxx    dxy]
                        [dxy    dyy]
        compute hessian:
                [dxx   dxy]         [00    01]
                            ====>
                [dxy   dyy]         [10    11]
        """
        point=[]
        direction=[]
        value=[]
        for x in range(0, self.col):
            for y in range(0, self.row):
                if dxy[y,x] > 0:
                    hessian = np.zeros((2,2))
                    hessian[0,0] = dxx[y,x]
                    hessian[0,1] = dxy[y,x]
                    hessian[1,0] = dxy[y,x]
                    hessian[1,1] = dyy[y,x]
                    # 计算矩阵的特征值和特征向量
                    _, eigenvalues, eigenvectors = cv2.eigen(hessian)
                    if np.abs(eigenvalues[0,0]) >= np.abs(eigenvalues[1,0]):
                        nx = eigenvectors[0,0]
                        ny = eigenvectors[0,1]
                    else:
                        nx = eigenvectors[1,0]
                        ny = eigenvectors[1,1]

                    # Taylor展开式分子分母部分,需要避免为0的情况
                    Taylor_numer = (dx[y, x] * nx + dy[y, x] * ny)
                    Taylor_denom = dxx[y,x]*nx*nx + dyy[y,x]*ny*ny + 2*dxy[y,x]*nx*ny
                    if Taylor_denom != 0:
                        T = -(Taylor_numer/Taylor_denom)
                        # Hessian矩阵最大特征值对应的特征向量对应于光条的法线方向
                        if np.abs(T*nx) <= threshold and np.abs(T*ny) <= threshold:
                            point.append((x,y))
                            direction.append((nx,ny))
                            value.append(np.abs(dxy[y,x]+dxy[y,x]))
        return point, direction, value

if __name__=="__main__":
    from pyzjr.visualize.core import Runcodes
    img = cv2.imread(r"E:\PythonProject\pyzjrPyPi\pyzjr\measure\fissure\0003.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    with Runcodes(description="Filter"):
        print(derivation(gray_img,1.1,1.1,"Filter"))
        # Filter: 0.01344 sec

    with Runcodes(description="Scharr"):
        print(derivation(gray_img, 1.1, 1.1))
        # Scharr: 0.00959 sec

    with Runcodes(description="Sobel"):
        print(derivation(gray_img, 1.1, 1.1,"Sobel"))
        # Sobel: 0.01820 sec

    img = cv2.imread(r"E:\PythonProject\pyzjrPyPi\pyzjr\measure\fissure\0003.png")

    steger = Steger(img, method="Sobel")
    img, newimage=steger.centerline(sigmaX=1.1,sigmaY=1.1)


    cv2.imwrite("result3.png", newimage)
    cv2.imwrite("result4.png", img)