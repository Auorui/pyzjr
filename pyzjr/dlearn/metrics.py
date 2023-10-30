# Dice系数 Dice Loss
# 混淆矩阵, 多分类TP, FN, FP, TN

import numpy as np
from torch import nn
import torch

def one_hot(labels, num_classes, device, dtype=torch.int64, eps=1e-6):
    """Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Examples:
        labels = torch.LongTensor([[[0, 1], [2, 0]]])
        one_hot(labels, num_classes=3, device=torch.device('cpu'), dtype=torch.int64)

        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])
    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"Input labels type is not a Tensor. Got {type(labels)}")

    if num_classes < 1:
        raise ValueError(f"The number of classes must be bigger than one. Got: {num_classes}")

    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

def get_onehot(labels, num_classes, dtype=torch.int64):
    """
    one-hot编码过程[b, H, W] -> [b, C, H, W]
    Args:
        label: b,h,w
        num_classes: 分类数
    Returns:
        b,num_classes,h,w
    """
    if num_classes < 1:
        raise ValueError(f"The number of classes must be bigger than one. Got: {num_classes}")

    def to_onehot(label, num_classes):
        return nn.functional.one_hot(label.to(dtype), num_classes)

    N = labels.size(0)
    onehot_labels = []
    for i in range(N):
        onehot = to_onehot(labels[i], num_classes)
        onehot_labels.append(onehot)

    return torch.stack(onehot_labels).permute(0, 3, 1, 2)


def Dice_coeff(pred, true, reduce_batch_first=False, epsilon=1e-6):
    """
    计算预测和目标的Dice系数的平均值
    :param pred: Tensor，预测的分割结果
    :param true: Tensor，真实的分割标签
    :param reduce_batch_first: bool，是否在批次维度上求平均，如果希望在整个批次上获得一个总体的Dice系数，
                               可以设置为 False。如果希望获得每个样本的Dice系数，并根据需要进行进一步的
                               处理，可以设置为 True。
    :param epsilon: float，平滑因子，避免分母为零
    :return: Tensor，Dice系数的平均值
    """
    assert pred.size() == true.size()
    if pred.dim() == 2 and reduce_batch_first:
        raise ValueError(f"[pyzjr]Request to reduce batches, but obtain tensors without batch dimensions")
    assert pred.dim() == 3 or not reduce_batch_first

    if pred.dim() == 2 or not reduce_batch_first:
        sum_dim = (-1, -2)
    else:
        sum_dim=(-1, -2, -3)

    inter = 2 * (pred * true).sum(dim=sum_dim)
    sets_sum = pred.sum(dim=sum_dim) + true.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_Dice_coeff(pred, true, reduce_batch_first=False, epsilon=1e-6):
    """
    计算多类别分割任务中所有类别的Dice系数的平均值
    [batch_size, num_classes, h, w] ——> [batch_size * num_classes, h, w]
    :param pred: Tensor，预测的分割结果
    :param true: Tensor，真实的分割标签
    :param reduce_batch_first: bool，是否在批次维度上求平均
    :param epsilon: float，平滑因子，避免分母为零
    :return: Tensor，所有类别的Dice系数的平均值
    """
    return Dice_coeff(pred.flatten(0, 1), true.flatten(0, 1), reduce_batch_first, epsilon)


def Dice_Loss(pred, true, multiclass=False):
    """
    计算Dice损失（目标是最小化），介于0和1之间
    :param pred: Tensor，预测的分割结果
    :param true: Tensor，真实的分割标签
    :param multiclass: bool，是否为多类别分割任务
    :return: Tensor，Dice损失
    """
    diceloss = multiclass_Dice_coeff if multiclass else Dice_coeff
    return 1 - diceloss(pred, true, reduce_batch_first=True)

class ConfusionMatrix(object):
    """For details:https://blog.csdn.net/m0_62919535/article/details/132893016"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, true, pred):
        t, p = true.flatten(), pred.flatten()
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=t.device)
        with torch.no_grad():
            k = (t >= 0) & (t < n)
            inds = n * t[k].to(torch.int64) + p[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    @property
    def ravel(self):
        """
        计算混淆矩阵的TN, FP, FN, TP
        支持二分类和多分类
        """
        h = self.mat.float()
        n = self.num_classes
        if n == 2:
            TP, FN, FP, TN = h.flatten()
            return TP, FN, FP, TN
        if n > 2:
            TP = h.diag()
            FN = h.sum(dim=1) - TP
            FP = h.sum(dim=0) - TP
            TN = torch.sum(h) - (torch.sum(h, dim=0) + torch.sum(h, dim=1) - TP)

            return TP, FN, FP, TN

    def compute(self):
        """
        主要在eval的时候使用,你可以调用ravel获得TN, FP, FN, TP, 进行其他指标的计算
        计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        计算每个类别的准确率
        计算每个类别预测与真实目标的iou,IoU = TP / (TP + FP + FN)
        """
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
            acc_global.item() * 100,
            ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100)

class ModelIndex():
    """For details:https://blog.csdn.net/m0_62919535/article/details/132926719"""
    def __init__(self,TP, FN, FP, TN, e=1e-5):
        if TP < 0 or FN < 0 or FP < 0 or TN < 0:
            raise ValueError("[pyzjr]:Data error, unable to perform calculation")
        self.TN = TN
        self.FP = FP
        self.FN = FN
        self.TP = TP
        self.e = e

    @property
    def Accuracy(self):
        """准确度是模型正确分类的样本数量与总样本数量之比"""
        score = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN + self.e)
        return score.mean(), score

    @property
    def Precision(self):
        """精确度衡量了正类别预测的准确性"""
        score = self.TP / (self.TP + self.FP + self.e)
        return score.mean(), score

    @property
    def Recall(self):
        """召回率衡量了模型对正类别样本的识别能力"""
        score = self.TP / (self.TP + self.FN + self.e)
        return score.mean(), score

    @property
    def IOU(self):
        """表示模型预测的区域与真实区域之间的重叠程度"""
        score = self.TP / (self.TP + self.FP + self.FN + self.e)
        return score.mean(), score

    @property
    def F1Score(self):
        """F1分数是精确度和召回率的调和平均数"""
        _, p = self.Precision
        _, r = self.Recall
        score = (2 * p * r) / (p + r + self.e)
        return score.mean(), score

    @property
    def Specificity(self):
        """特异性是指模型在负类别样本中的识别能力"""
        score = self.TN / (self.TN + self.FP + self.e)
        return score.mean(), score

    @property
    def FP_rate(self):
        """False Positive Rate,假阳率是模型将负类别样本错误分类为正类别的比例"""
        score = self.FP / (self.FP + self.TN + self.e)
        return score.mean(), score

    @property
    def FN_rate(self):
        """False Negative Rate,假阴率是模型将正类别样本错误分类为负类别的比例"""
        score = self.FN / (self.FN + self.TP + self.e)
        return score.mean(), score

    @property
    def Qualityfactor(self):
        """品质因子综合考虑了召回率和特异性"""
        _, r = self.Recall
        _, s = self.Specificity
        score = r+s-1
        return score.mean(), score

def NP_CFMatrix(a, b, n):
    """
    numpy形式的混淆矩阵
    a是转化成一维数组的标签，形状(H×W,)；
    b是转化成一维数组的预测结果，形状(H×W,)
    """
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)

# 需要进行测试,暂时没有投入实际使用中
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def accuracy_all_classes(output, target):
    """
    计算所有类别的正确率。
    """
    batch_size = target.size(0)
    _, pred = output.max(1)
    correct = pred.eq(target)
    accuracy = correct.float().sum() * 100. / batch_size

    return accuracy.item()


def Bessel(xi: list):
    """贝塞尔公式"""
    xi_array = np.array(xi)
    x_average = np.mean(xi_array)
    squared_diff = (xi_array - x_average) ** 2
    variance = squared_diff / (len(xi)-1)
    bessel = np.sqrt(variance)

    return bessel


if __name__ == "__main__":
    # 混淆矩阵测试代码
    true_labels = torch.tensor([0, 1, 2, 0, 1, 2])  # 真实标签
    predicted_labels = torch.tensor([0, 1, 1, 0, 2, 1])  # 预测结果
    conf = ConfusionMatrix(3)
    conf.update(true_labels,predicted_labels)
    TP, FN, FP, TN = conf.ravel

    index = ModelIndex(TP, FN, FP, TN)
    print(index.Precision)
    print(index.Accuracy)
    print(index.F1Score)