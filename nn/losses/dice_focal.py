import torch
import torch.nn as nn
import torch.nn.functional as F
from pyzjr.nn.losses.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

class DiceLoss(nn.Module):
    """
    Dice Loss 测量预测的和目标的二进制分割掩码之间的不相似性。它被计算为1减去Dice系数，Dice系数是重叠的度量
    在预测区域和目标区域之间。

    Args:
        mode (str): Loss 模式，可选 'binary', 'multiclass' 或 'multilabel'。
        reduction (str, optional): 指定对输出应用的缩减方式。
            可选值为 'none', 'mean', 或 'sum'。默认为 'mean'。

    Returns:
        torch.Tensor: The Dice Loss between input and target.
    """
    def __init__(self, mode=MULTILABEL_MODE, ignore_index=None, reduction='mean', eps=1e-5):
        super(DiceLoss, self).__init__()
        self.mode = mode
        self.eps = eps
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        if self.ignore_index:
            input = input[:,:self.ignore_index,...]
            target = target[:, :self.ignore_index, ...]

        if self.mode == BINARY_MODE:
            input = torch.sigmoid(input)
            intersection = torch.sum(input * target)
            union = torch.sum(input) + torch.sum(target)
        elif self.mode == MULTICLASS_MODE:
            input = torch.softmax(input, dim=1)
            target_one_hot = F.one_hot(target, num_classes=input.shape[1]).permute(0, 3, 1, 2).float()
            intersection = torch.sum(input * target_one_hot, dim=(2, 3))
            union = torch.sum(input, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
        elif self.mode == MULTILABEL_MODE:
            intersection = torch.sum(input * target, dim=(1,))
            union = torch.sum(input, dim=(1,)) + torch.sum(target, dim=(1,))

        dice_loss = 1 - (2 * intersection + self.eps) / (union + self.eps)
        return self.aggregate_loss(dice_loss)

    def aggregate_loss(self, focal_loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

class FocalLoss(nn.Module):
    """
    Focal Loss 用于解决类别不平衡问题，通过缩小易分类的类别的损失来关注难分类的类别。依据公式实现。

    Args:
        mode (str): Loss 模式，可选 'binary', 'multiclass' 或 'multilabel'。
        alpha (float, optional): 控制易分类的类别的权重，大于1表示增加权重，小于1表示减小权重。默认为1.
        gamma (float, optional): 控制难分类的类别的损失的下降速度，大于0表示下降较慢，小于0表示下降较快。默认为2.
        reduction (str, optional): 指定对输出应用的缩减方式。
            可选值为 'none', 'mean', 或 'sum'。默认为 'mean'。
    """
    def __init__(self,
                 mode: str = MULTILABEL_MODE,
                 alpha: float = 1,
                 gamma: float = 2,
                 ignore_index: int = None,
                 reduction: str = 'mean',
                 ):
        super(FocalLoss, self).__init__()
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        self.mode = mode
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.ignore_index:
            input = input[:,:self.ignore_index,...]
            target = target[:, :self.ignore_index, ...]
        if self.mode == BINARY_MODE:
            input = torch.sigmoid(input)
            ce_loss = F.binary_cross_entropy_with_logits(input, target.float(), reduction='none')
            class_weights = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - class_weights) ** self.gamma * ce_loss
        elif self.mode == MULTICLASS_MODE:
            ce_loss = F.cross_entropy(input, target.long(), reduction='none')
            class_weights = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - class_weights) ** self.gamma * ce_loss
        elif self.mode == MULTILABEL_MODE:
            ce_loss = F.binary_cross_entropy_with_logits(input, target.float(), reduction='none')
            class_weights = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - class_weights) ** self.gamma * ce_loss

        return self.aggregate_loss(focal_loss)

    def aggregate_loss(self, focal_loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

class DiceFocalLoss(nn.Module):
    """
    DiceFocalLoss结合了Dice Loss和Focal Loss两种损失函数，用于解决图像分割任务中的类别不平衡和边界模糊的问题。
    在计算损失时，综合考虑了模型预测结果与真实标签的 Dice Loss 和 Focal Loss 。

    Args:
        ignore_index (int, optional): 需要忽略的标签索引，如果设置了该参数，则计算损失时会忽略这些标签。默认为None。
        reduction (str, optional): 损失值的缩减模式，可选值为'mean'（求平均）、'sum'（求和）或'none'（不缩减）。默认为'mean'。
        eps (float, optional): 用于数值稳定性的小值。默认为1e-5。
        lambda_dice (float, optional): Dice Loss的权重系数。默认为1.0。
        lambda_focal (float, optional): Focal Loss的权重系数。默认为1.0。

    Examples::
        >>> criterion = DiceFocalLoss(ignore_index=0, reduction='mean', lambda_dice=0.8, lambda_focal=0.2)
        >>> input_data = torch.rand((4, 2, 16, 16), dtype=torch.float32)
        >>> target_data = torch.randint(0, 2, (4, 2, 16, 16), dtype=torch.float32)
        >>> loss = criterion(input_data, target_data)
        >>> print("DiceFocal Loss:", loss.item())

    Returns:
        torch.Tensor: 计算得到的DiceFocal Loss值。
    """
    def __init__(self,
                 mode=MULTILABEL_MODE ,
                 ignore_index=None,
                 reduction='mean',
                 eps=1e-5,
                 lambda_dice=1.0,
                 lambda_focal=1.0):
        super(DiceFocalLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.dice = DiceLoss(
            mode,
            ignore_index=ignore_index,
            reduction='none',
        )
        self.focal = FocalLoss(
            mode,
            ignore_index=ignore_index,
            reduction='none',
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal

    def forward(self, input, target):
        dice_loss = self.dice(input, target)
        focal_loss = self.focal(input, target)
        total_loss = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss

        return self.aggregate_loss(total_loss)

    def aggregate_loss(self, total_loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        elif self.reduction == 'none':
            return total_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

class JaccardLoss(nn.Module):
    """
    Jaccard Loss 用于衡量预测分割与目标分割之间的相似度。它是通过计算预测区域和目标区域的交集与并集之间的比值来衡量的。

    Args:
        mode (str): Loss 模式，可选 'binary', 'multiclass' 或 'multilabel'。
        ignore_index (int, optional): 忽略的类别索引，默认为 None。
        reduction (str, optional): 指定对输出应用的缩减方式。
            可选值为 'none', 'mean', 或 'sum'。默认为 'mean'。
        eps (float, optional): 避免除以零的小量值。默认为 1e-5。
    """
    def __init__(self, mode=MULTILABEL_MODE, ignore_index=None, reduction='mean', eps=1e-5):
        super(JaccardLoss, self).__init__()
        self.mode = mode
        self.eps = eps
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        if self.ignore_index:
            input = input[:, :self.ignore_index, ...]
            target = target[:, :self.ignore_index, ...]

        if self.mode == BINARY_MODE:
            input = torch.sigmoid(input)
            intersection = torch.sum(input * target, dim=(1,))
            union = torch.sum(input, dim=(1,)) + torch.sum(target, dim=(1,))
        elif self.mode == MULTICLASS_MODE:
            input = torch.softmax(input, dim=1)
            target_one_hot = F.one_hot(target, num_classes=input.shape[1]).permute(0, 3, 1, 2).float()
            intersection = torch.sum(input * target_one_hot, dim=(2, 3))
            union = torch.sum(input, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
        elif self.mode == MULTILABEL_MODE:
            intersection = torch.sum(input * target, dim=(1,))
            union = torch.sum(input, dim=(1,)) + torch.sum(target, dim=(1,))

        jaccard_loss = 1.0 - (intersection + self.eps) / (union - intersection + self.eps)
        return self.aggregate_loss(jaccard_loss)

    def aggregate_loss(self, jaccard_loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'mean':
            return jaccard_loss.mean()
        elif self.reduction == 'sum':
            return jaccard_loss.sum()
        elif self.reduction == 'none':
            return jaccard_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

if __name__=="__main__":
    # Binary Mode
    criterion_binary_focal = FocalLoss(mode='binary', alpha=1, gamma=2, reduction='mean')
    criterion_binary_dice = DiceLoss(mode=BINARY_MODE, reduction='mean')
    criterion_binary_iou = JaccardLoss(mode=BINARY_MODE, reduction='mean')
    input_binary = torch.rand((2, 1, 16, 16))
    target_binary = torch.randint(0, 2, (2, 1, 16, 16))
    loss_binary_focal = criterion_binary_focal(input_binary, target_binary)
    loss_binary_dice = criterion_binary_dice(input_binary, target_binary)
    loss_binary_iou = criterion_binary_iou(input_binary, target_binary)
    print(input_binary.shape, target_binary.shape)
    print("Binary Focal Loss:", loss_binary_focal.item())
    print("Binary Dice Loss:", loss_binary_dice.item())
    print("Binary IOU Loss:", loss_binary_iou.item())

    # Multiclass Mode
    criterion_multiclass_focal = FocalLoss(mode=MULTICLASS_MODE, alpha=1, gamma=2, reduction='mean')
    criterion_multiclass_dice = DiceLoss(mode=MULTICLASS_MODE, reduction='mean')
    criterion_multiclass_iou = JaccardLoss(mode=MULTICLASS_MODE, reduction='mean')
    input_multiclass = torch.randn((2, 3, 16, 16))
    target_multiclass = torch.randint(0, 3, (2, 16, 16))
    loss_multiclass_focal = criterion_multiclass_focal(input_multiclass, target_multiclass)
    loss_multiclass_dice = criterion_multiclass_dice(input_multiclass, target_multiclass)
    loss_multiclass_iou = criterion_multiclass_iou(input_multiclass, target_multiclass)
    print(input_multiclass.shape, target_multiclass.shape)
    print("Multiclass Focal Loss:", loss_multiclass_focal.item())
    print("Multiclass Dice Loss:", loss_multiclass_dice.item())
    print("Multiclass IOU Loss:", loss_multiclass_iou.item())

    # Multilabel Mode
    criterion_multilabel_focal = FocalLoss(mode=MULTILABEL_MODE, alpha=1, gamma=2, reduction='mean')
    criterion_multilabel_dice = DiceLoss(mode=MULTILABEL_MODE, reduction='mean')
    criterion_multilabel_iou = JaccardLoss(mode=MULTILABEL_MODE, reduction='mean')
    input_multilabel = torch.randn((2, 3, 16, 16))
    target_multilabel = torch.randint(0, 2, (2, 3, 16, 16))
    loss_multilabel_focal = criterion_multilabel_focal(input_multilabel, target_multilabel)
    loss_multilabel_dice = criterion_multilabel_dice(input_multilabel, target_multilabel)
    loss_multilabel_iou = criterion_multilabel_iou(input_multilabel, target_multilabel)
    print(input_multilabel.shape, target_multilabel.shape)
    print("Multilabel Focal Loss:", loss_multilabel_focal.item())
    print("Multilabel Dice Loss:", loss_multilabel_dice.item())
    print("Multilabel IOU Loss:", loss_multilabel_iou.item())
