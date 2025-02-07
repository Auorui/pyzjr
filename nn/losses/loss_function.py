"""
Copyright (c) 2024, Auorui.
All rights reserved.
time 2024-01-25
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["L1Loss", "L2Loss", "BCELoss",
           "Joint2loss",
           "MCCLoss"]

class L1Loss(nn.Module):
    """
    L1损失，也称为平均绝对误差（MAE），测量预测输出中的每个元素与目标或地面实况中的相应元素之间的平均绝对差。
    在数学上，它表示为预测值和目标值之间差异的绝对值的平均值。与L2损耗相比，L1损耗对异常值不那么敏感。依据公式实现。
    Args:
        input (torch.Tensor): The predicted output.
        target (torch.Tensor): The target or ground truth.
        reduction (str, optional): Specifies the reduction to apply to the output.
            Options are 'none', 'mean', or 'sum'. Default is 'mean'.
    Examples::
        >>> criterion1 = nn.L1Loss()
        >>> criterion2 = L1Loss()
        >>> input_data=torch.Tensor([2, 3, 4, 5])
        >>> target_data=torch.Tensor([4, 5, 6, 7])
        >>> loss1 = criterion1(input_data, target_data)  # tensor(2.)
        >>> loss2 = criterion2(input_data, target_data)  # tensor(2.)
    Returns:
        torch.Tensor: The L1 loss between input and target.
    """
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, input, target):
        loss = torch.mean(torch.abs(input - target))
        return loss

class L2Loss(nn.Module):
    """
    L2损失，也称为均方误差（MSE），测量预测输出中的每个元素与目标或地面实况中的相应元素之间的平均平方差。
    在数学上，它表示为预测值和目标值之间差异的平方的平均值。相比于L1损耗，L2损耗对异常值更敏感。依据公式实现。
    在torch当中是MSELoss
    Args:
        input (torch.Tensor): The predicted output.
        target (torch.Tensor): The target or ground truth.
    Examples::
        >>> criterion1 = nn.MSELoss()
        >>> criterion2 = L2Loss()
        >>> input_data=torch.Tensor([2, 3, 4, 5])
        >>> target_data=torch.Tensor([4, 5, 6, 7])
        >>> loss1 = criterion1(input_data, target_data)  # tensor(4.)
        >>> loss2 = criterion2(input_data, target_data)  # tensor(4.)

    Returns:
        torch.Tensor: The L2 loss between input and target.
    """
    def __init__(self, eps=1e-3):
        super(L2Loss, self).__init__()
        self.esp = eps

    def forward(self, input, target):
        loss = torch.mean(torch.pow(input - target, 2) + self.esp)
        return loss

class BCELoss(nn.Module):
    """
    二元交叉熵损失（Binary Cross Entropy Loss），也称为对数损失。
    用于测量预测输出中的每个元素与目标或地面实况中的相应元素之间的对数概率差异。依据公式实现。
    Args:
        input (torch.Tensor): The predicted output.Map to (0,1) through sigmoid function.
        target (torch.Tensor): The target or ground truth.
        reduction (str, optional): Specifies the reduction to apply to the output.
            Options are 'none', 'mean', or 'sum'. Default is 'mean'.

    Examples::
        >>> criterion1 = nn.BCELoss()
        >>> criterion2 = BCELoss()
        >>> input_data = torch.randn((5,))
        >>> target_data = torch.randint(0, 2, (5,), dtype=torch.float32)
        >>> loss1 = criterion1(torch.sigmoid(input_data), target_data)
        >>> loss2 = criterion2(input_data, target_data)
        >>> print("PyTorch BCELoss:", loss1.item())
        >>> print("Custom BCELoss:", loss2.item())

    Returns:
        torch.Tensor: The binary cross entropy loss between input and target.
    """
    def __init__(self, ignore_index=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.ignore_index:
            input = input[:,:self.ignore_index,...]
            target = target[:, :self.ignore_index, ...]
        input = torch.sigmoid(input)
        loss = - (target * torch.log(input) + (1 - target) * torch.log(1 - input))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')


class Joint2loss(nn.Module):
    """
    联合损失函数, 传入两个损失函数
        >>> criterion1 = FocalLoss()
        >>> criterion2 = DiceLoss()
        >>> joint_loss = Joint2loss(criterion1, criterion2, alpha=0.7, reduction='mean')
        >>> input_tensor = torch.rand((1,2,16,16), dtype=torch.float32)
        >>> target_tensor = torch.randn((1,2,16,16), dtype=torch.float32)
        >>> loss = joint_loss(input_tensor, target_tensor)
        >>> print("Joint Loss:", loss.item())
    """
    def __init__(self, *args, alpha, beta=None, ignore_index=None, reduction='mean'):
        super(Joint2loss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.beta = beta if beta is not None else 1-self.alpha
        self.criterion_1, self.criterion_2 = args

    def forward(self, input, target):
        if self.ignore_index:
            input = input[:, :self.ignore_index,...]
            target = target[:, :self.ignore_index, ...]
        loss_1 = self.criterion_1(input, target)
        loss_2 = self.criterion_2(input, target)
        loss = self.alpha * loss_1 + self.beta * loss_2
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

class MCCLoss(nn.Module):
    def __init__(self, eps: float = 1e-5, threshold: float =.5):
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
        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
        Args:
            y_pred (torch.Tensor): model prediction of shape (N, H, W) or (N, 1, H, W)
            y_true (torch.Tensor): ground truth labels of shape (N, H, W) or (N, 1, H, W)

        Returns:
            torch.Tensor: loss value (1 - mcc)
        """
        y_pred = (y_pred >= self.threshold).float()
        bs = y_true.shape[0]

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        tp = torch.sum(torch.mul(y_pred, y_true)) + self.eps
        tn = torch.sum(torch.mul((1 - y_pred), (1 - y_true))) + self.eps
        fp = torch.sum(torch.mul(y_pred, (1 - y_true))) + self.eps
        fn = torch.sum(torch.mul((1 - y_pred), y_true)) + self.eps

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(torch.add(tp, fp) * torch.add(tp, fn) * torch.add(tn, fp) * torch.add(tn, fn))

        mcc = torch.div(numerator.sum(), denominator.sum())
        loss = 1.0 - mcc

        return loss


class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', max_val=1):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.max_val = max_val

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
        psnr = 10 * torch.log10((self.max_val ** 2) / (mse + 1e-8))
        return -self.loss_weight * torch.mean(psnr)

if __name__=="__main__":
    batch_size = 2
    height, width = 4, 4
    y_true = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    y_pred = torch.rand((batch_size, 1, height, width))

    mcc_loss = MCCLoss()
    loss = mcc_loss(y_pred, y_true)
    print(f'MCC Loss: {loss.item()}')

    y_pred = torch.randn(1, 1, 2, 2)
    y_true = torch.randn(1, 1, 2, 2)
    psnr_loss_fn = PSNRLoss(loss_weight=1.0)
    loss = psnr_loss_fn(y_pred, y_true)
    print("PSNR Loss:", loss.item())