"""
Copyright (c) 2025, Auorui.
All rights reserved.

This module is used for reading and displaying images and videos. The image contains
OpenCV and PIL, and the loading method for the video is OpenCV.
"""
from .blend import blend_mask_on_image, OverlayPng
from .videos import VideoCap, FPS
from .matplot import matplotlib_patch, plot_images
from .readio import pil_to_ndarray, ndarray_to_pil, tensor_to_ndarray, ndarray_to_tensor, \
    read_image, write_image, show_image, stacked_images
from .colorspace import to_gray, rgb2bgr, bgr2rgb, to_hsv, hsv2rgb, hsv2bgr, create_palette, get_colors
from .draw import DrawPolygon, DrawCornerRectangle
from .text import AddText, PutMultiLineText, PutMultiLineCenteredText, PutBoxText, PutRectangleText
