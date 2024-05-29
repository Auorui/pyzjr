"""
This module provides 3D neural networks for DSCNet, UNet, and VNet

Copyright (c) 2024, Auorui.
All rights reserved.
time 2024-04-28 The last weekend of April.

https://github.com/HiLab-git/SSL4MIS/tree/master
"""
from .basic_unet import BasicUNet3D
from .vnet import VNet3D
from .dscnet import DSCNet3D
from .transunet import TransUnet3D

__all__ = ["BasicUNet3D", "VNet3D", "DSCNet3D", "TransUnet3D"]