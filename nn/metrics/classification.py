"""
Copyright (c) 2023, Auorui.
All rights reserved.

分类指标有二分类, 多分类, 多标签.
由于二分类和多分类的任务较多, 所以一下函数仅仅涉及这二者.
用于 分类 的指标检测
                 | Predicted Positive | Predicted Negative |
Actual Positive  |        TP          |         FN         |
Actual Negative  |        FP          |         TN         |
Time: 2024-01-27
"""
import torch
import numpy as np

class cls_matrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.matrix = np.zeros((self.num_classes, self.num_classes))
        if self.num_classes <= 1:
            raise ValueError("Argument num_classes needs to be > 1")

    def update(self, pred, true):
        # 在派生类中实现逻辑
        pass

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes))

    @property
    def get_matrix(self):
        return self.matrix

class BinaryConfusionMatrix(cls_matrix):
    """
    二分类混淆矩阵类, num_classes默认就为2
    Example:
        >>> confusion_matrix = BinaryConfusionMatrix()
        >>> pred = torch.tensor([0.4, 0.5, 0.5, 0.7, 1, 0.8])
        >>> true = torch.tensor([0, 1, 1, 0, 0, 1])
        >>> confusion_matrix.update(pred, true)
        >>> matrix = confusion_matrix.get_matrix
        >>> print(matrix, confusion_matrix.ravel())
    """
    def __init__(self, threshold=.5):
        super().__init__(num_classes=2)
        self.threshold = threshold
        self.num_classes = 2
        self.reset()

    def update(self, pred, true):
        if torch.is_tensor(pred):
            pred = pred.cpu().detach().numpy()
        if torch.is_tensor(true):
            true = true.cpu().detach().numpy()

        pred = (pred >= self.threshold).astype(int)
        # true = np.where(true >= self.threshold, 1, 0)

        true_positive = np.sum((pred == 1) & (true == 1))
        false_positive = np.sum((pred == 1) & (true == 0))
        true_negative = np.sum((pred == 0) & (true == 0))
        false_negative = np.sum((pred == 0) & (true == 1))

        self.matrix[0, 0] += true_negative
        self.matrix[0, 1] += false_positive
        self.matrix[1, 0] += false_negative
        self.matrix[1, 1] += true_positive

    def ravel(self):
        TP, FN, FP, TN = self.matrix.flatten()
        return TP, FN, FP, TN

class MulticlassConfusionMatrix(cls_matrix):
    """
    多分类混淆矩阵类, num_classes大于2
    Example:
        >>> confusion_matrix = MulticlassConfusionMatrix(num_classes=5, reduction='mean')
        >>> pred = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        >>> true = torch.tensor([0, 1, 2, 3, 4, 1, 2, 3, 4, 0])
        >>> confusion_matrix.update(pred, true)
        >>> matrix = confusion_matrix.get_matrix
        >>> print(matrix, confusion_matrix.ravel())
    """
    def __init__(self, num_classes, reduction='mean'):
        super().__init__(num_classes=num_classes)
        self.reduction = reduction
        self.reset()
        self.threshold = .5

    def update(self, pred, true):
        if torch.is_tensor(pred):
            pred = pred.cpu().detach().numpy()
        if torch.is_tensor(true):
            true = true.cpu().detach().numpy()

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.matrix[i, j] += np.sum((true == i) & (pred == j))

    def ravel(self):
        """
        计算混淆矩阵的TN, FP, FN, TP
        支持二分类和多分类
        """
        h = self.matrix.astype(float)
        TP = np.diag(h)
        FN = np.sum(h, axis=1) - TP
        FP = np.sum(h, axis=0) - TP
        TN = np.sum(h) - (np.sum(h, axis=0) + np.sum(h, axis=1) - TP)
        if self.reduction == 'mean':
            return np.mean(TP), np.mean(FN), np.mean(FP), np.mean(TN)
        elif self.reduction == 'sum':
            return np.sum(TP), np.sum(FN), np.sum(FP), np.sum(TN)
        elif self.reduction == 'none':
            return TP, FN, FP, TN

class ConfusionMatrixs(cls_matrix):
    """
    结合多分类与二分类两种情况, 用法与这二者相同
    """
    def __init__(self, num_classes, reduction='mean'):
        super().__init__(num_classes)
        self.num_classes = num_classes
        self.reduction = reduction
        if self.num_classes == 2:
            self.conf = BinaryConfusionMatrix(threshold=0.5)
        elif self.num_classes > 2:
            self.conf = MulticlassConfusionMatrix(self.num_classes, self.reduction)

    def update(self, pred, true):
        self.conf.update(pred, true)

    def ravel(self):
        TP, FN, FP, TN = self.conf.ravel()
        return TP, FN, FP, TN

    @property
    def get_matrix(self):
        return self.conf.get_matrix

def calculate_metrics(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    metrics = np.zeros((4,))

    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FN = np.sum(confusion_matrix[i, :]) - TP
        FP = np.sum(confusion_matrix[:, i]) - TP
        TN = np.sum(confusion_matrix) - (TP + FN + FP)
        metrics[0] += TP
        metrics[1] += FN
        metrics[2] += FP
        metrics[3] += TN
    return metrics

def generate_class_weights(num_classes):
    # 类别权重,每个类别平均分配权重
    total_weight = 1.0
    r = 1 if num_classes < 10 else 2
    class_weights = [round(total_weight / num_classes, r)] * (num_classes - 1)
    class_weights.append(round(total_weight - sum(class_weights), r))
    return np.array(class_weights)

class ModelIndex():
    """For details:https://blog.csdn.net/m0_62919535/article/details/132926719
        >>> true_labels = torch.tensor([0, 1, 2, 0, 1, 2])  # 真实标签
        >>> predicted_labels = torch.tensor([0, 1, 1, 0, 2, 1])  # 预测结果
        >>> conf = MulticlassConfusionMatrix(3)
        >>> conf.update(true_labels, predicted_labels)
        >>> TP, FN, FP, TN = conf.ravel()
        >>> index = ModelIndex(TP, FN, FP, TN)
        >>> print(index)

    reduction mean, sum, none, weighted
    """
    def __init__(self,TP, FN, FP, TN, reduction='mean', headline="ModelIndex", esp=1e-5):
        self.TN = TN
        self.FP = FP
        self.FN = FN
        self.TP = TP
        self.reduction = reduction
        self.headline = headline
        self.e = esp
        if self.reduction == 'weighted' or self.reduction == 'none':
            self.num_classes = self.TP.shape[0]
            self.weight = generate_class_weights(self.num_classes)

    @property
    def Accuracy(self):
        """准确度是模型正确分类的样本数量与总样本数量之比"""
        score = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN + self.e)
        if self.reduction == 'mean':
            return score.mean()
        elif self.reduction == 'sum':
            return score.sum()
        elif self.reduction == 'none':
            return score
        elif self.reduction == 'weighted':
            return (score * self.weight).sum()

    @property
    def Precision(self):
        """精确度衡量了正类别预测的准确性"""
        score = self.TP / (self.TP + self.FP + self.e)
        if self.reduction == 'mean':
            return score.mean()
        elif self.reduction == 'sum':
            return score.sum()
        elif self.reduction == 'none':
            return score
        elif self.reduction == 'weighted':
            return (score * self.weight).sum()

    @property
    def Recall(self):
        """召回率衡量了模型对正类别样本的识别能力"""
        score = self.TP / (self.TP + self.FN + self.e)
        if self.reduction == 'mean':
            return score.mean()
        elif self.reduction == 'sum':
            return score.sum()
        elif self.reduction == 'none':
            return score
        elif self.reduction == 'weighted':
            return (score * self.weight).sum()

    @property
    def IOU(self):
        """表示模型预测的区域与真实区域之间的重叠程度"""
        score = self.TP / (self.TP + self.FP + self.FN + self.e)
        if self.reduction == 'mean':
            return score.mean()
        elif self.reduction == 'sum':
            return score.sum()
        elif self.reduction == 'none':
            return score
        elif self.reduction == 'weighted':
            return (score * self.weight).sum()

    @property
    def F1Score(self):
        """F1分数是精确度和召回率的调和平均数"""
        p = self.Precision
        r = self.Recall
        score = (2 * p * r) / (p + r + self.e)
        if self.reduction == 'mean':
            return score.mean()
        elif self.reduction == 'sum':
            return score.sum()
        elif self.reduction == 'none':
            return score
        elif self.reduction == 'weighted':
            return (score * self.weight).sum()

    @property
    def Specificity(self):
        """特异性是指模型在负类别样本中的识别能力"""
        score = self.TN / (self.TN + self.FP + self.e)
        if self.reduction == 'mean':
            return score.mean()
        elif self.reduction == 'sum':
            return score.sum()
        elif self.reduction == 'none':
            return score
        elif self.reduction == 'weighted':
            return (score * self.weight).sum()

    @property
    def FP_rate(self):
        """False Positive Rate,假阳率是模型将负类别样本错误分类为正类别的比例"""
        score = self.FP / (self.FP + self.TN + self.e)
        if self.reduction == 'mean':
            return score.mean()
        elif self.reduction == 'sum':
            return score.sum()
        elif self.reduction == 'none':
            return score
        elif self.reduction == 'weighted':
            return (score * self.weight).sum()

    @property
    def FN_rate(self):
        """False Negative Rate,假阴率是模型将正类别样本错误分类为负类别的比例"""
        score = self.FN / (self.FN + self.TP + self.e)
        if self.reduction == 'mean':
            return score.mean()
        elif self.reduction == 'sum':
            return score.sum()
        elif self.reduction == 'none':
            return score
        elif self.reduction == 'weighted':
            return (score * self.weight).sum()

    @property
    def Qualityfactor(self):
        """品质因子综合考虑了召回率和特异性"""
        r = self.Recall
        s = self.Specificity
        score = r+s-1
        if self.reduction == 'mean':
            return score.mean()
        elif self.reduction == 'sum':
            return score.sum()
        elif self.reduction == 'none':
            return score
        elif self.reduction == 'weighted':
            return (score * self.weight).sum()

    @property
    def Dice(self):
        """Dice系数更多的是用于分割方面"""
        score = 2 * self.TP / (2 * self.TP + self.FN + self.FP + self.e)
        if self.reduction == 'mean':
            return score.mean()
        elif self.reduction == 'sum':
            return score.sum()
        elif self.reduction == 'none':
            return score
        elif self.reduction == 'weighted':
            return (score * self.weight).sum()

    def __str__(self):
        acc_mean = self.Accuracy
        prec_mean = self.Precision
        recall_mean = self.Recall
        iou_mean = self.IOU
        f1_mean = self.F1Score
        spec_mean = self.Specificity
        fp_rate_mean = self.FP_rate
        fn_rate_mean = self.FN_rate
        qf_mean = self.Qualityfactor
        dice_mean = self.Dice
        output_str = f"\033[94m{self.headline}:\n" \
                     f"\t\033[94mClass\tAccuracy\tPrecision    Recall     IOU  \tF1Score\t    Specificity\t   " \
                     f"FP Rate\t  FN Rate\t  Quality Factor\t  Dice\n"
        if self.reduction == 'mean' or self.reduction == 'sum' or self.reduction == 'weighted':
            output_str += (f"\t\033[94mall  \t{acc_mean:.5f}      {prec_mean:.5f}    {recall_mean:.5f}   "
                    f"{iou_mean:.5f}   {f1_mean:.5f}       {spec_mean:.5f}      {fp_rate_mean:.5f}    "
                    f"{fn_rate_mean:.5f}        {qf_mean:.5f}        {dice_mean:.5f}")

        else:
            for i in range(self.TP.shape[0]):
                output_str += f"\t\033[94m{i}\t    {acc_mean[i]:.5f}\t     {prec_mean[i]:.5f}\t{recall_mean[i]:.5f}\t  " \
                              f"{iou_mean[i]:.5f}\t{f1_mean[i]:.5f}\t      {spec_mean[i]:.5f}\t   {fp_rate_mean[i]:.5f}\t  " \
                              f"{fn_rate_mean[i]:.5f}\t     {qf_mean[i]:.5f}\t    {dice_mean[i]:.5f}\n"
            output_str += (f"\t\033[94mmean   \t{acc_mean.mean():.5f}      {prec_mean.mean():.5f}    {recall_mean.mean():.5f}   "
             f"{iou_mean.mean():.5f}   {f1_mean.mean():.5f}       {spec_mean.mean():.5f}      {fp_rate_mean.mean():.5f}    "
             f"{fn_rate_mean.mean():.5f}        {qf_mean.mean():.5f}        {dice_mean.mean():.5f}\n")
            output_str += (f"\t\033[94mweight  {(acc_mean* self.weight).sum():.5f}      {(prec_mean* self.weight).sum():.5f}    {(recall_mean* self.weight).sum():.5f}   "
                           f"{(iou_mean * self.weight).sum():.5f}   {(f1_mean* self.weight).sum():.5f}       {(spec_mean* self.weight).sum():.5f}      {(fp_rate_mean* self.weight).sum():.5f}    "
                           f"{(fn_rate_mean* self.weight).sum():.5f}        {(qf_mean* self.weight).sum():.5f}        {(dice_mean* self.weight).sum():.5f}")
        return output_str

    def get_Index(self):
        metrics_data = {
            'Accuracy': self.Accuracy,
            'Precision': self.Precision,
            'Recall': self.Recall,
            'IOU': self.IOU,
            'F1Score': self.F1Score,
            'Specificity': self.Specificity,
            'FP_rate': self.FP_rate,
            'FN_rate': self.FN_rate,
            'Qualityfactor': self.Qualityfactor,
            'Dice': self.Dice,
        }
        return metrics_data


if __name__=="__main__":
    confusion_matrix = ConfusionMatrixs(num_classes=3, reduction='none')
    true = torch.tensor([1, 1, 1, 0, 0, 0, 2, 2, 2, 2])
    pred = torch.tensor([1, 0, 0, 0, 2, 1, 0, 0, 2, 2])
    confusion_matrix.update(pred, true)
    TP, FN, FP, TN = confusion_matrix.ravel()
    print(TP, FN, FP, TN)
    print(confusion_matrix.get_matrix)

    index = ModelIndex(TP, FN, FP, TN, reduction='mean')
    print(index)






































