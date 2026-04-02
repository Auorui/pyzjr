"""
Copyright (c) 2025, Auorui.
All rights reserved.

This module is used for reading and displaying images and videos. The image contains
OpenCV and PIL, and the loading method for the video is OpenCV.
"""
from .videos import VideoCap, FPS
from .matplot import imshowplt, StackedpltV1, StackedpltV2, matplotlib_patch
from .readio import read_bgr, read_gray, read_rgb, read_tensor, read_url, read_image, imwrite, display,\
    StackedImagesV1, StackedImagesV2