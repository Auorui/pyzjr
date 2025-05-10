"""
Copyright (c) 2025, Auorui.
All rights reserved.

对图像的标签进行处理, 包括灰度、二值化、骨架提取、轮廓提取、距离变换等等
"""
from .predeal import (uint2single, single2uint, clip, unique, create_rectmask, binarization,
                      approximate, ceilfloor, bool2mask, up_low, inpaint_defect, convert_mask,
                      cvt8png, auto_canny, mask_foreground_move)
from .skeleton_extraction import medial_axis_mask, skeletonizes, thinning, read_skeleton
from .distance_transform import cv_distance, chamfer, fast_marching
from .statistical_pixels import (count_nonzero, count_white, count_zero, unique, incircleV1,
                                incircleV2, outcircle)
from .contour import (SearchOutline, check_points_in_contour, getContours, foreground_contour_length,
                      sort_contours, label_contour, gradientOutline, drawOutline)
from .steger import Steger
