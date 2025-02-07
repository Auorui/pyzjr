import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    """
    交叉熵损失（Cross Entropy Loss）用于多分类问题。
    用于测量预测输出和目标分布之间的交叉熵。依据公式实现。
    Args:
        input (torch.Tensor): The predicted output (logits).
        target (torch.Tensor): The target or ground truth (class labels).
        reduction (str, optional): Specifies the reduction to apply to the output.
            Options are 'none', 'mean', or 'sum'. Default is 'mean'.

    Examples::
        >>> criterion1 = nn.CrossEntropyLoss()
        >>> criterion2 = CrossEntropyLoss()
        >>> input_data = torch.randn((3, 5), requires_grad=True)
        >>> target_data = torch.randint(0, 5, (3,))
        >>> loss1 = criterion1(input_data, target_data)
        >>> loss2 = criterion2(input_data, target_data)
        >>> print("PyTorch CrossEntropyLoss:", loss1.item())
        >>> print("Custom CrossEntropyLoss:", loss2.item())

    Returns:
        torch.Tensor: The cross entropy loss between input and target.
    """
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return nn.NLLLoss(reduction=self.reduction)(F.log_softmax(input, dim=1), target)

class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing."""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, input, target):
        logprobs = F.log_softmax(input, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


if __name__=="__main__":
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = CrossEntropyLoss()
    criterion3 = LabelSmoothingCrossEntropy()
    input_data = torch.randn((3, 5), requires_grad=True)
    target_data = torch.randint(0, 5, (3,))
    loss1 = criterion1(input_data, target_data)
    loss2 = criterion2(input_data, target_data)
    loss3 = criterion3(input_data, target_data)
    print("PyTorch CrossEntropyLoss:", loss1.item())
    print("Custom CrossEntropyLoss:", loss2.item())
    print("Custom LabelSmoothingCrossEntropy:", loss3.item())