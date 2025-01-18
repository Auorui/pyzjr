"""
Copyright (c) 2023, Auorui.
All rights reserved.

The following citation imports roughly follow the sorting by year
"""
# 1990s
from .lenet import LeNet

# 2010s
from .alexnet import AlexNet, ZFNet
from .vgg import VGG, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .googlenet import GoogLeNet
from .resnet import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152
from .squeezenet import SqueezeNet, squeezenet1_0, squeezenet1_1
from .darknet import darknet19, darknet53
from .mobilenet import MobileNetV1, MobileNetV2, MobileNetV3, MobileNetV3_Large, MobileNetV3_Small
from .densenet import DenseNet, densenet121, densenet161, densenet169, densenet201
from .mnasnet import MNASNet, mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
from .shufflenet import ShuffleNetV1, ShuffleNetV2, shufflenet_v1_g1, shufflenet_v1_g2, \
    shufflenet_v1_g3, shufflenet_v1_g4, shufflenet_v1_g8, shufflenet_v2_x0_5, shufflenet_v2_x1_0, \
    shufflenet_v2_x1_5, shufflenet_v2_x2_0
from .xception import Xception
from .drn import DRN, DRN_A, drn_a_50, drn_c_26, drn_c_42, drn_c_58, drn_d_22, drn_d_24, drn_d_38, \
    drn_d_40, drn_d_54, drn_d_56, drn_d_105, drn_d_107

# 2020s
from .conv2former import conv2former_n, conv2former_t, conv2former_s, conv2former_l, conv2former_b
from .ghostnet import GhostNet, GGhostRegNet, ghostnetv1, ghostnetv2, g_ghost_regnetx_002, \
    g_ghost_regnetx_004, g_ghost_regnetx_006, g_ghost_regnetx_008, g_ghost_regnetx_016, \
    g_ghost_regnetx_032, g_ghost_regnetx_040, g_ghost_regnetx_064, g_ghost_regnetx_080, \
    g_ghost_regnetx_120, g_ghost_regnetx_160, g_ghost_regnetx_320
from .regnet import RegNet, regnetx_002, regnetx_004, regnetx_006, regnetx_008, \
    regnetx_016, regnetx_032, regnetx_040, regnetx_064, regnetx_080, regnetx_120, regnetx_160, regnetx_320
from .fasternet import FasterNet, fasternet_t0, fasternet_t1, fasternet_t2, fasternet_s, \
    fasternet_m, fasternet_l

# vision Transformer
from .vision_transformer import VisionTransformer, vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14
from .swin_transformer import SwinTransformer, swin_t, swin_s, swin_b, swin_l

se_resnet18 = lambda num_classes, **kwargs: resnet18(num_classes=num_classes, use_se=True, **kwargs)
se_resnet34 = lambda num_classes, **kwargs: resnet34(num_classes=num_classes, use_se=True, **kwargs)
se_resnet50 = lambda num_classes, **kwargs: resnet50(num_classes=num_classes, use_se=True, **kwargs)
se_resnet101 = lambda num_classes, **kwargs: resnet101(num_classes=num_classes, use_se=True, **kwargs)
se_resnet152 = lambda num_classes, **kwargs: resnet152(num_classes=num_classes, use_se=True, **kwargs)



