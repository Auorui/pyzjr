#: Loss multilabel mode suppose you are solving multi-**label** segmentation task.
#: That mean you have *C = 1..N* classes which pixels are labeled as **1**,
#: classes are not mutually exclusive and each class have its own *channel*,
#: pixels in each channel which are not belong to class labeled as **0**.
#: Target mask shape - (N, C, H, W), model output mask shape (N, C, H, W).
from .loss_utils import JointLoss
from .loss_function import (
    L1Loss,
    L2Loss,
    BCELoss,
    MCCLoss,
    DiceLoss,
    FocalLoss,
    JaccardLoss,
    CrossEntropyLoss,
)
