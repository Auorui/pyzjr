"""
Copyright (c) 2023, Auorui.
All rights reserved.

* VOCs and RGBs: These are color lists used to provide a predefined set of colors.
VOCs may be colors used for visual object category (VOC) datasets, while RGBs contain
a set of RGB colors. Colormap, random_color, getPalette, and createcolored_canvas: These
are a series of functions that provide functions such as color mapping, random color
generation, palette retrieval, and creating canvases filled with specified colors.
* ColorFind: is a class or function used to find specific colors in an image.
* Grayscale, rgb2bgr, bgr2rgb, bgr2hsv, hsv2bgr, rgb2hsv, hsv2rgb: These are functions
used for color space conversion, including grayscale conversion and the mutual conversion
between RGB and HSV.
OverlayPng, putBoxText, coreRect, cvt2Center, cvt2Corner, AddText, DrawPolygon,
DrawBboxPolygon: These are functions used to draw specific shapes, text, and more
on an image.
* HexColors: is a class or function used to handle hexadecimal color values.
"""
from .colormap import VOC_COLORS, RGB_COLORS, colormap, random_color, \
    getPalette, create_colored_canvas
from .findcolor import ColorFind
from .colorspace import grayscale, rgb2bgr, bgr2rgb, bgr2hsv, hsv2bgr, rgb2hsv, hsv2rgb
from .cvplot import OverlayPng, putBoxText, cornerRect, cvt2Center,\
    cvt2Corner, AddText, DrawPolygon, DrawBboxPolygon
from .hex import HexColors
