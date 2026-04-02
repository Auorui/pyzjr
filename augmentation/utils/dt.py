## distance transform ---> dt
import cv2
import numpy as np
from skimage.morphology import medial_axis
from scipy.ndimage import distance_transform_edt

def euclidean_dt(image, maskSize=3):
    """Euclidean distance transform"""
    return cv2.distanceTransform(image, distanceType=cv2.DIST_L2, maskSize=maskSize)

def chamfer_dt(image, kernel_weights=None):
    """Chamfer distance transformation"""
    if kernel_weights is None:
        kernel_weights = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=np.float32)

    # Convert image to float32 and normalize to 0-1 range
    normalized_img = image.astype(np.float32) / 255.0
    return cv2.filter2D(normalized_img, cv2.CV_32F, kernel_weights)

def fast_marching_dt(image):
    """Fast-Marching Distance Transform"""
    _, medial_axis_image = medial_axis(image > 0, return_distance=True)   # 使用medial_axis函数计算中轴线
    dist_transform = distance_transform_edt(medial_axis_image)
    return dist_transform

if __name__=="__main__":
    image = np.zeros((10, 10), dtype=np.uint8)
    image[2:8, 2:8] = 255
    print(image)

    dist_transform = euclidean_dt(image)
    chamfer_dist_transform = chamfer_dt(image)
    fast_marching_dist_transform = fast_marching_dt(image)
    # 距离变换结果
    print("Distance Transform:")
    print(dist_transform)
    print("\nChamfer Distance Transform:")
    print(chamfer_dist_transform)
    print("\nFast-Marching Distance Transform:")
    print(fast_marching_dist_transform)