from .adaptivepooling import adaptive_pool2d, adaptive_catavgmax_pool2d, adaptive_avgmax_pool2d,\
    AdaptiveCatAvgMaxPool2d, AdaptiveAvgMaxPool2d, FastAdaptiveAvgPool2d, SelectAdaptivePool2d
from .standardpooling import MaxPool2d, AvgPool2d
from .haarwavelet import HWDownsampling
from .strideconv import StridedConv
from .strippooling import StripPooling
from .medianpooling import MedianPool2d

# Downsampling processing module

__all__ = ["adaptive_pool2d", "adaptive_avgmax_pool2d", "adaptive_catavgmax_pool2d",
           "AdaptiveAvgMaxPool2d", "AdaptiveCatAvgMaxPool2d", "FastAdaptiveAvgPool2d",
           "SelectAdaptivePool2d", "MaxPool2d", "AvgPool2d", "HWDownsampling", "StridedConv",
           "StripPooling", "MedianPool2d"]