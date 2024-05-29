import torch
import torch.nn as nn
from pyzjr.nn.losses._utils import (
    sigmoid_focal_loss_3d,
    softmax_focal_loss_3d,
)
import warnings

class FocalLoss3D(nn.Module):
    """
    3D Focal Loss实现，用于处理3D图像分割任务。

    Args:
        include_background (bool): 是否包含背景类在损失计算中。默认为True。
        gamma (float): Focal Loss中的聚焦参数。默认为2.0。
        alpha (list or None): 用于平衡正负样本的权重。默认为None，表示不使用类别权重。
        use_softmax (bool): 是否使用softmax函数进行多分类。默认为False，表示使用sigmoid进行二分类或多标签分类。
        reduction (str): 指定损失函数的归约方式，可选值有'none', 'mean', 'sum'。默认为'mean'，表示计算损失的平均值。
    Example:
        >>> focal = FocalLoss3D()
        >>> batch_size = 2
        >>> num_classes = 3
        >>> d = h = w = 64
        >>> predictions = torch.randint(0, num_classes, (batch_size, num_classes, d, h, w))
        >>> true_labels = torch.randint(0, num_classes, (batch_size, num_classes, d, h, w))
        >>> loss = focal(predictions, true_labels)
        >>> print(f"Loss from FocalLoss3D: {loss.item()}")
    """
    def __init__(self,
                 include_background=True,
                 gamma=2.0,
                 alpha=None,
                 use_softmax=False,
                 reduction='mean'):
        super(FocalLoss3D, self).__init__()
        self.include_background = include_background
        self.gamma = gamma
        self.alpha = alpha
        self.use_softmax = use_softmax
        self.reduction = reduction

    def forward(self, input, target):
        n_pred_ch = input.shape[1]
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]
        input = input.float()
        target = target.float()
        if target.shape != input.shape:
            raise ValueError(f"ground truth has different shape ({target.shape}) from input ({input.shape}),"
                             f"It may require one hot encoding")
        if self.use_softmax:
            if not self.include_background and self.alpha is not None:
                self.alpha = None
                warnings.warn("`include_background=False`, `alpha` ignored when using softmax.")
            loss = softmax_focal_loss_3d(input, target, self.gamma, self.alpha)
        else:
            loss = sigmoid_focal_loss_3d(input, target, self.gamma, self.alpha)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

class DiceLoss3D(nn.Module):
    """
    用于计算三维 Dice Loss 的 PyTorch 模块。
    注意“input”的轴N被期望为每个类的logits或概率，必须设置“sigmoid=True”或“softmax=True”
    Args:
        include_background (bool): 是否包括背景类，默认为 True。
        sigmoid (bool): 是否应用 sigmoid 函数，默认为 False。
        softmax (bool): 是否应用 softmax 函数，默认为 False。
        squared_pred (bool): 是否使用预测值的平方，默认为 False。
        jaccard (bool): 是否使用 Jaccard 损失（相当于 Dice Loss 的变体），默认为 False。
        reduction (str): 损失值的减少方式，可选值包括 "mean"、"sum" 和 "none"，默认为 "mean"。
        smooth_nr (float): 分子平滑参数，默认为 1e-5。
        smooth_dr (float): 分母平滑参数，默认为 1e-5。
    Example:
        >>> dice = DiceLoss3D()
        >>> batch_size = 2
        >>> num_classes = 3
        >>> d = h = w = 64
        >>> predictions = torch.randint(0, num_classes, (batch_size, num_classes, d, h, w))
        >>> true_labels = torch.randint(0, num_classes, (batch_size, num_classes, d, h, w))
        >>> loss = dice(predictions, true_labels)
        >>> print(f"Loss from DiceLoss3D: {loss.item()}")
    """
    def __init__(
            self,
            include_background=True,
            sigmoid=False,
            softmax=False,
            squared_pred=False,
            jaccard = False,
            reduction='mean',
            smooth_nr=1e-5,
            smooth_dr=1e-5,
    ):
        super(DiceLoss3D, self).__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape}),"
                                 f"It may require one hot encoding")

        reduce_axis = torch.arange(2, len(input.shape)).tolist()
        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            ground_o = torch.sum(target**2, dim=reduce_axis)
            pred_o = torch.sum(input**2, dim=reduce_axis)
        else:
            ground_o = torch.sum(target, dim=reduce_axis)
            pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

        if self.reduction == "mean":
            return torch.mean(f)
        elif self.reduction == "sum":
            return torch.sum(f)
        elif self.reduction == "none":
            return f
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

class DiceFocalLoss3D(nn.Module):
    """
  用于结合 Dice Loss 和 Focal Loss 的三维损失函数模块。

    Args:
        include_background (bool): 是否包含背景类在损失计算中。默认为 True。
        sigmoid (bool): 是否应用 sigmoid 函数到预测值。默认为 False。
        softmax (bool): 是否应用 softmax 函数到预测值。默认为 False。多类别要改为 True。
        squared_pred (bool): 是否使用预测值的平方版本。仅用于 Dice Loss。
        jaccard (bool): 是否计算 Jaccard Index（软 IoU）而不是 Dice。默认为 False。
        reduction (str): 损失值的减少方式，可选值为 "mean"、"sum" 和 "none"。默认为 "mean"。
        smooth_nr (float): 分子平滑参数，默认为 1e-5。
        smooth_dr (float): 分母平滑参数，默认为 1e-5。
        gamma (float): Focal Loss 中的聚焦参数。默认为 2.0。
        lambda_dice (float): Dice Loss 的权重值。默认为 1.0。
        lambda_focal (float): Focal Loss 的权重值。默认为 1.0。
    Args:

    Example:
        >>> dice_focal = DiceFocalLoss3D()
        >>> batch_size = 2
        >>> num_classes = 3
        >>> d = h = w = 64
        >>> predictions = torch.randint(0, num_classes, (batch_size, num_classes, d, h, w))
        >>> true_labels = torch.randint(0, num_classes, (batch_size, num_classes, d, h, w))
        >>> loss = dice_focal(predictions, true_labels)
        >>> print(f"Loss from DiceFocalLoss3D: {loss.item()}")
    """
    def __init__(
            self,
            include_background=True,
            sigmoid=False,
            softmax=False,
            squared_pred=False,
            jaccard=False,
            reduction="mean",
            smooth_nr=1e-5,
            smooth_dr=1e-5,
            gamma=2.0,
            lambda_dice=1.0,
            lambda_focal=1.0,
    ):
        super(DiceFocalLoss3D, self).__init__()
        self.dice = DiceLoss3D(
            include_background=include_background,
            sigmoid=sigmoid,
            softmax=softmax,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction='mean',
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )
        self.focal = FocalLoss3D(
            include_background=include_background,
            gamma=gamma,
            use_softmax=softmax,
            reduction='mean',
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.reduction = reduction

    def forward(self, input, target):
        dice_loss = self.dice(input, target)
        focal_loss = self.focal(input, target)
        total_loss = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss

        if self.reduction == "mean":
            return torch.mean(total_loss)
        elif self.reduction == "sum":
            return torch.sum(total_loss)
        elif self.reduction == "none":
            return total_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')


if __name__=="__main":
    focal = DiceFocalLoss3D()
    batch_size = 2
    num_classes = 3
    d = h = w = 64
    predictions = torch.randint(0, num_classes, (batch_size, num_classes, d, h, w))
    true_labels = torch.randint(0, num_classes, (batch_size, num_classes, d, h, w))
    loss = focal(predictions, true_labels)
    print(f"Loss from FocalLoss3D: {loss.item()}")