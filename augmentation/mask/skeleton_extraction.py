import cv2
import numpy as np
import skimage.filters as filters
from skimage.morphology import skeletonize, dilation, disk, medial_axis
from skimage import io, morphology
from scipy import ndimage as ndi
from pyzjr.augmentation.mask.predeal import bool2mask, binarization, G123P_LUT, G123_LUT

def medial_axis_mask(image):
    """
    The input must be a binary graph to obtain the axis image within it
    """
    result = medial_axis(image)
    return bool2mask(result).astype(np.uint8)

def skeletonizes(image, size=3, structuring=cv2.MORPH_RECT):
    """通过反复腐蚀和膨胀操作，逐步剥离物体的外层像素，最终提取出物体的中心线（骨架）"""
    area = image.shape[0] * image.shape[1]
    skeleton = np.zeros(image.shape, dtype="uint8")
    elem = cv2.getStructuringElement(structuring, (size, size))
    while True:
        eroded = cv2.erode(image, elem)
        temp = cv2.dilate(eroded, elem)
        temp = cv2.subtract(image, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image = eroded.copy()
        if area == area - cv2.countNonZero(image):
            break
    return skeleton

def thinning(image, max_num_iter=None):
    """
    Perform morphological thinning of a binary image.

    Parameters
    ----------
    image : binary (M, N) ndarray
        The image to be thinned.
    max_num_iter : int, number of iterations, optional
        Regardless of the value of this parameter, the thinned image
        is returned immediately if an iteration produces no change.
        If this parameter is specified it thus sets an upper bound on
        the number of iterations performed.

    Notes
    -----
    This algorithm works by making multiple passes over the image,
    removing pixels matching a set of criteria designed to thin
    connected regions while preserving eight-connected components and
    2 x 2 squares. In each of the two sub-iterations the algorithm
    correlates the intermediate skeleton image with a neighborhood mask,
    then looks up each neighborhood in a lookup table indicating whether
    the central pixel should be deleted in that sub-iteration.
    """
    # convert image to uint8 with values in {0, 1}
    skel = np.asanyarray(image, dtype=bool).astype(np.uint8)
    # neighborhood mask
    mask = np.array([[ 8,  4,   2],
                     [16,  0,   1],
                     [32, 64, 128]], dtype=np.uint8)
    max_num_iter = max_num_iter or np.inf
    num_iter = 0
    n_pts_old, n_pts_new = np.inf, np.sum(skel)

    while n_pts_old != n_pts_new and num_iter < max_num_iter:
        n_pts_old = n_pts_new
        for lut in [G123_LUT, G123P_LUT]:
            # correlate image with neighborhood mask
            N = ndi.correlate(skel, mask, mode='constant')
            # take deletion decision from this subiteration's LUT
            D = np.take(lut, N)
            # perform deletion
            skel[D] = 0

        n_pts_new = np.sum(skel)
        num_iter += 1

    return bool2mask(skel.astype(bool)).astype(np.uint8)

def read_skeleton(image_path, read_type="cv2", skeleton_type="skeleton"):
    """
    Reads an image from a given path and generates its skeleton or medial axis transform image.

    image_path: str
        The path to the image file.
    read_type: str, optional
        Specifies the method to read and process the image. Supports "cv2" for OpenCV or "skimage" for scikit-image. Defaults to "cv2".
    skeleton_type: str, optional
        Specifies the type of skeleton to generate. Supports "skeleton" for thinning skeleton or "medial_axis" for medial axis transform. Defaults to "skeleton".
    return: numpy.ndarray
        The processed skeleton or medial axis transform image, represented in grayscale.
    """
    binary = skeletons = None
    if read_type == "cv2":
        image = cv2.imread(image_path)
        binary = binarization(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=1)
        binary = cv2.medianBlur(binary, 5)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    elif read_type == "skimage":
        image = io.imread(image_path, as_gray=True)
        thresh = filters.threshold_otsu(image)
        binary = image > thresh
        binary = dilation(binary, disk(3))
        binary = filters.median(binary, footprint=morphology.disk(5))
        selem = morphology.disk(3)
        binary = morphology.closing(binary, selem)

    if skeleton_type == "skeleton":
        skeletons = skeletonize(binary)
        skeletons = skeletons.astype(np.uint8) * 255
    elif skeleton_type == "medial_axis":
        skeletons = medial_axis_mask(binary)
    elif skeleton_type == "thin":
        skeletons = thinning(binary)
    return skeletons


if __name__=="__main__":
    img_path = r"E:\PythonProject\pyzjrPyPi\pyzjr\measure\crack\0005.png"

    image = cv2.imread(img_path)
    medial_image = medial_axis_mask(binarization(image))
    sk_image = skeletonizes(binarization(image))
    skio_image = read_skeleton(img_path, "skimage", "skeleton")
    skcv_image = read_skeleton(img_path, "cv2", "skeleton")
    skio_medial_image = read_skeleton(img_path, "skimage", "medial_axis")
    skcv_medial_image = read_skeleton(img_path, "cv2", "medial_axis")
    skio_image_thin = read_skeleton(img_path, "skimage", "thin")
    skcv_image_thin = read_skeleton(img_path, "cv2", "thin")
    print("skio_image", np.unique(skio_image))
    print("skcv_image", np.unique(skcv_image))
    print("skio_medial_image", np.unique(skio_medial_image))
    print("skcv_medial_image", np.unique(skcv_medial_image))
    print("skio_image_thin", np.unique(skio_image_thin))
    print("skcv_image_thin", np.unique(skcv_image_thin))

    # cv2.imshow("medial_image", medial_image)
    # cv2.imshow("sk_image", sk_image)
    cv2.imshow("skio_medial_image", skio_medial_image)
    cv2.imshow("skio_image", skio_image)
    cv2.imshow("skcv_image", skcv_image)
    cv2.imshow("skcv_medial_image", skcv_medial_image)
    cv2.imshow("skio_image_thin", skio_image_thin)
    cv2.imshow("skcv_image_thin", skcv_image_thin)
    cv2.waitKey(0)