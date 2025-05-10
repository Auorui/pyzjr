"""
Copyright (c) 2024, Auorui.
All rights reserved.
time 2024-01-25
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class L1Loss(nn.Module):
    """
    L1 Loss (MAE Loss)
        Measure the average absolute difference between each element in the predicted output
    and the corresponding element in the target.
    Args:
        input (torch.Tensor): The predicted output.
        target (torch.Tensor): The target or ground truth.
    Returns:
        torch.Tensor: The L1 loss between input and target.
    """
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(torch.abs(y_pred - y_true))
        return loss

class L2Loss(nn.Module):
    """
    L2 Loss (MSE Loss)
        Measure the average squared difference between each element in the predicted
    output and the corresponding element in the target。

    Args:
        input (torch.Tensor): The predicted output.
        target (torch.Tensor): The target or ground truth.

    Returns:
        torch.Tensor: The L2 loss between input and target.
    """
    def __init__(self, eps=1e-3):
        super(L2Loss, self).__init__()
        self.esp = eps

    def forward(self, y_pred, y_true):
        loss = torch.mean(torch.pow(y_pred - y_true, 2) + self.esp)
        return loss

class BCELoss(nn.Module):
    """
    BCELoss (Binary Cross Entropy Loss)
        Used to measure the logarithmic probability difference between each element in
    the predicted output and the corresponding element in the target, implemented according to the formula.

    Args:
        input (torch.Tensor): The predicted output.Map to (0,1) through sigmoid function.
        target (torch.Tensor): The target or ground truth.

    Returns:
        torch.Tensor: The binary cross entropy loss between input and target.
    """
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        loss = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        return loss.mean()


class MCCLoss(nn.Module):
    def __init__(self, eps: float = 1e-5, threshold: float = .5):
        """Compute Matthews Correlation Coefficient Loss for image segmentation task.
        It only supports binary mode.

        Args:
            eps (float): Small epsilon to handle situations where all the samples in the dataset belong to one class

        Reference:
            https://github.com/kakumarabhishek/MCC-Loss
        """
        super().__init__()
        self.eps = eps
        self.threshold = threshold

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute MCC loss
        MCC = (TP * TN - FP * FN) / sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN))
        Args:
            y_pred (torch.Tensor): model prediction of shape (N, H, W) or (N, 1, H, W)
            y_true (torch.Tensor): ground truth labels of shape (N, H, W) or (N, 1, H, W)

        Returns:
            torch.Tensor: loss value (1 - mcc)
        """
        y_pred = (y_pred >= self.threshold).float()

        y_true = y_true.view(y_true.shape[0], 1, -1)
        y_pred = y_pred.view(y_true.shape[0], 1, -1)

        tp = torch.sum(torch.mul(y_pred, y_true)) + self.eps
        tn = torch.sum(torch.mul((1 - y_pred), (1 - y_true))) + self.eps
        fp = torch.sum(torch.mul(y_pred, (1 - y_true))) + self.eps
        fn = torch.sum(torch.mul((1 - y_pred), y_true)) + self.eps

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(torch.add(tp, fp) * torch.add(tp, fn) * torch.add(tn, fp) * torch.add(tn, fn))

        mcc = torch.div(numerator.sum(), denominator.sum())
        loss = 1.0 - mcc
        return loss

class CrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss:
        Used to measure the cross entropy between the predicted output and the target
    distribution.
    CE Loss = nn.NLLLoss(reduction=self.reduction)(F.log_softmax(y_pred, dim=1), y_true)

    Args:
        input (torch.Tensor): The predicted output (logits).
        target (torch.Tensor): The target or ground truth (class labels).
        reduction (str, optional): Specifies the reduction to apply to the output.
            Options are 'none', 'mean', or 'sum'. Default is 'mean'.

    Returns:
        torch.Tensor: The cross entropy loss between input and target.
    """
    def __init__(self, reduction='mean', ignore_index: int = -100, weight=None):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, y_pred, y_true):
        b, c, h, w = y_pred.shape
        if self.weight is None:
            self.weight = torch.ones(c, dtype=torch.float32)

        return F.cross_entropy(
            y_pred, y_true,
            weight=self.weight,
            ignore_index=self.ignore_index
        )

class DiceLoss(nn.Module):
    """
    Dice Loss:
        Measure the dissimilarity between the predicted and target binary segmentation
    masks. It is calculated as 1 minus the Dice coefficient, which is a measure of overlap between the predicted and target regions.

    Args:
        reduction (str, optional): Specify the reduction method applied to the output.
            The optional values are 'none', 'mean', or 'sum'. The default is' mean '.

    Returns:
        torch.Tensor: The Dice Loss between input and target.
    """
    def __init__(self, ignore_index=None, eps=1e-5):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, y_pred, y_true):
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true)
        dice_loss = 1 - (2 * intersection + self.eps) / (union + self.eps)
        return dice_loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss:
        Used to solve the problem of class imbalance, focusing on difficult to classify
    categories by reducing the loss of easy to classify categories。

    Args:
        alpha (float, optional): Control the weight of easy to classify categories, with a value
            greater than 1 indicating an increase in weight and a value less than 1 indicating a decrease in weight. The default is 1
        gamma (float, optional): Control the rate of loss reduction for difficult to classify categories, where a value greater than 0
            indicates a slower decrease and a value less than 0 indicates a faster decrease. The default is 2
        reduction (str, optional): Specify the reduction method applied to the output.
            The optional values are 'none', 'mean', or 'sum'. The default is' mean '.
    """
    def __init__(self,
                 alpha: float = .5,
                 gamma: float = 2,
                 ignore_index: int = -100,
                 reduction: str = 'none',
                 ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, y_pred, y_true) -> torch.Tensor:
        logpt = -nn.CrossEntropyLoss(weight=None, ignore_index=self.ignore_index,
                                     reduction=self.reduction)(y_pred, y_true)
        pt = torch.exp(logpt)
        logpt *= self.alpha
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()


class JaccardLoss(nn.Module):
    """
    Jaccard Loss:
        Used to measure the similarity between predicted segmentation and target segmentation.
    It is measured by calculating the ratio between the intersection and union of the predicted
    area and the target area.

    Args:
        eps (float, optional): Avoid small values divided by zero. The default is 1e-5.
    """
    def __init__(self, eps=1e-5):
        super(JaccardLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true)
        jaccard_loss = 1.0 - intersection / (union - intersection + self.eps)
        return jaccard_loss.mean()



if __name__ == "__main__":
    input_data = torch.Tensor([2, 3, 4, 5])
    target_data = torch.Tensor([4, 5, 6, 7])
    loss1 = nn.L1Loss()(input_data, target_data)
    loss2 = L1Loss()(input_data, target_data)
    print(loss1, loss2)
    loss1 = nn.MSELoss()(input_data, target_data)
    loss2 = L2Loss()(input_data, target_data)
    print(loss1, loss2)
    input_data = torch.randn((5,))
    target_data = torch.randint(0, 2, (5,), dtype=torch.float32)
    loss3 = nn.BCELoss()(torch.sigmoid(input_data), target_data)
    loss4 = BCELoss()(input_data, target_data)
    print(loss3, loss4)
    y_pred = torch.rand(4, 3, 5, 5)
    y_true = torch.randint(0, 2, (4, 3, 5, 5)).float()
    loss5 = MCCLoss()(y_pred, y_true)
    print(loss5)
    loss6 = nn.CrossEntropyLoss()(y_pred, y_true)
    loss7 = CrossEntropyLoss()(y_pred, y_true)
    print(loss6, loss7)
    print(y_pred.shape, y_true.shape)
    loss8 = DiceLoss()(y_pred, y_true)
    loss9 = FocalLoss()(y_pred, y_true)
    loss10 = JaccardLoss()(y_pred, y_true)
    print(loss8, loss9, loss10)