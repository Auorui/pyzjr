from .transforms import ToTensor, CenterCrop
import torchvision.transforms as T

import pyzjr.Z as Z

IMAGENET_MEAN = Z.IMAGENET_MEAN
IMAGENET_STD  = Z.IMAGENET_STD

def classify_transforms(size=224):
    # Previously used on YOLOv3
    # T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose([CenterCrop(size), ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])