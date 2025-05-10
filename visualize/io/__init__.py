"""
Copyright (c) 2025, Auorui.
All rights reserved.

This module is used for reading and displaying images and videos. The image contains
OpenCV and PIL, and the loading method for the video is OpenCV.
"""
from .video_use import VideoCap
from .matplot import imshowplt, StackedpltV1, StackedpltV2, matplotlib_patch
from .readio import read_bgr, read_gray, read_rgb, read_tensor, read_image, imwrite, display,\
    StackedImagesV1, StackedImagesV2, imattributes
from .imtensor import (
    image_to_bchw, bchw_to_image, tensor_to_image, image_to_tensor, imagelist_to_tensor
)