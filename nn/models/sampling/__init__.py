from .adaptivepooling import adaptive_pool2d, adaptive_catavgmax_pool2d, adaptive_avgmax_pool2d,\
    AdaptiveCatAvgMaxPool2d, AdaptiveAvgMaxPool2d, FastAdaptiveAvgPool2d, SelectAdaptivePool2d
from .pool import MaxPool2d, AvgPool2d, StridedPool2d, MedianPool2d

from .haarwavelet import HWDownsampling
from .strippooling import StripPooling

# Downsampling processing module

__all__ = [
    "adaptive_pool2d",
    "adaptive_avgmax_pool2d",
    "adaptive_catavgmax_pool2d",
    "AdaptiveAvgMaxPool2d",
    "AdaptiveCatAvgMaxPool2d",
    "FastAdaptiveAvgPool2d",
    "SelectAdaptivePool2d",
    "MaxPool2d",
    "AvgPool2d",
    "HWDownsampling",
    "StridedPool2d",
    "StripPooling",
    "MedianPool2d"
]