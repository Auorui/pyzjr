import cv2
import string
from pyzjr.augmentation import SearchOutline, BinaryImg, count_nonzero, drawOutline
from pyzjr.measure.crack.crack_type import infertype
from pyzjr.measure.pixel import incircle, foreground_contour_length, get_each_crack_areas

def crop_crack_according_to_bbox(mask, Bboxing):
    """根据边界框,进行裁剪,并存入列表中"""
    cropped_cracks = []
    for bbox in Bboxing:
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]
        cropped_crack = mask[y1:y2, x1:x2]
        cropped_cracks.append(cropped_crack)
    return cropped_cracks

def crack_labels(cracknums):
    """创建一个关于A-Z的标签"""
    labels = []
    for i in range(cracknums):
        label = string.ascii_uppercase[i]
        labels.append(label)
    return labels

def DetectCrack(mask, contours_algorithm=1, draw_contour=True):
    mask_copy = mask.copy()
    all_mask_areas = count_nonzero(mask)      # 总面积

    gray_image = cv2.cvtColor(mask_copy, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cracktype = infertype(gray_image)          # 判断整体类型
    crack_info = {
        'crack_type': cracktype,
        'crack_numbers': 0,
        'total_areas': all_mask_areas,
        'total_lengths': None,
        'avg_width': None,
    }
    if contours_algorithm == 0:
        contours_arr, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    elif contours_algorithm == 1:
        contours_arr = SearchOutline(thresh)
    result, each_width = incircle(gray_image, contours_arr)
    if cracktype == 'Mesh':
        return result, crack_info, each_width
    else:
        each_area, Bboxing, result = get_each_crack_areas(result, thresh)
        crack_numbers = len(each_area)
        labels = crack_labels(crack_numbers)
        cropped_cracks_img = crop_crack_according_to_bbox(mask, Bboxing)
        each_info = []
        total_length = 0
        total_area = 0
        total_width = 0
        for i, crack_img in enumerate(cropped_cracks_img):
            crack_img = BinaryImg(crack_img)
            contour_lengths, all_length = foreground_contour_length(crack_img)  # 这里的总长度是轮廓长度, 故 / 2
            each_info.append(
                [labels[i], all_length / 2, each_area[labels[i]], each_width[labels[i]]]
            )

            total_length += all_length / 2
            total_area += each_area[labels[i]]
            total_width += each_width[labels[i]]
        avg_width = total_width / crack_numbers
        crack_info["crack_numbers"] = crack_numbers
        crack_info["total_areas"] = total_area
        crack_info["total_lengths"] = total_length
        crack_info["avg_width"] = avg_width
        if draw_contour:
            drawOutline(result, contours_arr)
        return result, crack_info, each_info


if __name__=="__main__":
    path = r'D:\PythonProject\pyzjrPyPi\models_img\1604.png'
    cv_image = cv2.imread(path)
    result, crack_info, each_info = DetectCrack(cv_image)
    print(crack_info, "\n", each_info)
    cv2.imwrite("ss1.png", result)
    cv2.imshow("ss", result)
    cv2.waitKey(0)