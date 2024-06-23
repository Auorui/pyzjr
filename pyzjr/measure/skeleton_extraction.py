# Skeleton extraction
"""
skeletoncv、sketionio均是使用zhang-suen骨架算法（手写算法不佳），sketion_medial_axis采用的是中轴提取
下面的膨胀、中值滤波（滤波效果中最佳）、闭运算是经过实验进行的组合，针对自己的数据集可以先尝试如果效果不佳再采取实验
"""
import cv2
import numpy as np
import skimage.filters as filters
from skimage.morphology import skeletonize, dilation, disk
from skimage import io, morphology
from pyzjr.augmentation.mask_ops import BinaryImg, medial_axis_mask

def skeletoncv(single_pic_path):
    image = cv2.imread(single_pic_path)
    binary = BinaryImg(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.medianBlur(binary, 5)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.medianBlur(binary, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    skeleton = skeletonize(binary / 255)
    skeleton = (skeleton * 255).astype(np.uint8)
    return skeleton

def sketionio(single_pic_path):
    image = io.imread(single_pic_path, as_gray=True)
    thresh = filters.threshold_otsu(image)
    binary = image > thresh

    binary = dilation(binary, disk(3))
    binary = filters.median(binary, footprint=morphology.disk(5))
    binary = dilation(binary, disk(2))
    binary = filters.median(binary, footprint=morphology.disk(5))

    selem = morphology.disk(3)
    binary = morphology.closing(binary, selem)

    skeleton = skeletonize(binary)
    skeleton = skeleton.astype(np.uint8) * 255
    return skeleton

def sketion_medial_axis(single_pic_path):
    image = cv2.imread(single_pic_path)
    binary = BinaryImg(image)

    binary = dilation(binary, disk(2))
    binary = filters.median(binary, footprint=morphology.disk(5))
    binary = dilation(binary, disk(2))
    binary = filters.median(binary, footprint=morphology.disk(5))
    selem = morphology.disk(3)
    binary = morphology.closing(binary, selem)
    binary = medial_axis_mask(binary)
    return binary.astype(np.uint8)


if __name__=="__main__":
    from pyzjr.augmentation.mask_ops import unique
    from pyzjr.augmentation.mask_ops import count_nonzero
    path = r'D:\PythonProject\pyzjrPyPi\models_img\1604.png'
    image_cv = skeletoncv(path)
    print(unique(image_cv))
    print(image_cv.shape)
    print(count_nonzero(image_cv))

    image_io = sketionio(path)
    print(unique(image_io))
    print(image_io.shape)

    image_axis = sketion_medial_axis(path)
    print(unique(image_axis))
    print(image_axis.shape)

    cv2.imshow("test skeleton function cv", image_cv)
    cv2.imshow("test skeleton function io", image_io)
    cv2.imshow("test skeleton function medial axis", image_axis)
    cv2.waitKey(0)