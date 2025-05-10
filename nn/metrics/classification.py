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
import warnings
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

class class_matrix(object):
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

class BinaryConfusionMatrix(class_matrix):
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
        # 多分类任务中通常不计算TN,因为对于每个类别,其他所有类别都可以被视为“负类”,这使得TN的计算变得复杂且不直观
        false_negative = np.sum((pred == 0) & (true == 1))

        self.matrix[0, 0] += true_negative
        self.matrix[0, 1] += false_positive
        self.matrix[1, 0] += false_negative
        self.matrix[1, 1] += true_positive

    def ravel(self):
        TP, FN, FP, TN = self.matrix.flatten()
        return TP, FN, FP, TN

class MulticlassConfusionMatrix(class_matrix):
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

class ConfusionMatrixs(class_matrix):
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

    def plot_confusion_matrix(self, save_path="./class_confusionmatrix.png"):
        cm = self.get_matrix.astype(int)
        if self.num_classes == 2:
            class_names=['Negative','Positive']
        else:
            class_names=[f'Class {i}' for i in range(self.num_classes)]
        matplotlib.use('TkAgg')
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names,cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()

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

class ClassificationIndex():
    """
    二分类与多分类
    For details:https://blog.csdn.net/m0_62919535/article/details/132926719
    """
    def __init__(self,TP, FN, FP, TN, reduction='mean', headline="ClassificationIndex", esp=1e-5):
        self.esp = esp
        self.reduction = reduction
        self.headline = headline
        self.TN, self.FP, self.FN, self.TP = TN, FP, FN, TP

    def _safe_divide(self, dividend, divisor):
        return dividend / (divisor + self.esp)

    def apply_reduction_to_score(self, score):
        if self.reduction == 'mean':
            return score.mean()
        elif self.reduction == 'sum':
            return score.sum()
        elif self.reduction == 'none':
            return score

    @property
    def accuracy(self):
        """准确度是模型正确分类的样本数量与总样本数量之比"""
        score = self._safe_divide(self.TP + self.TN, self.TP + self.TN + self.FP + self.FN)
        return self.apply_reduction_to_score(score)

    @property
    def precision(self):
        """精确度衡量了正类别预测的准确性"""
        score = self._safe_divide(self.TP, self.TP + self.FP)
        return self.apply_reduction_to_score(score)

    @property
    def recall(self):
        """召回率衡量了模型对正类别样本的识别能力"""
        score = self._safe_divide(self.TP, self.TP + self.FN)
        return self.apply_reduction_to_score(score)

    @property
    def iou(self):
        """表示模型预测的区域与真实区域之间的重叠程度"""
        score = self._safe_divide(self.TP, self.TP + self.FP + self.FN)
        return self.apply_reduction_to_score(score)

    @property
    def f1score(self):
        """F1分数是精确度和召回率的调和平均数"""
        p = self._safe_divide(self.TP, self.TP + self.FP)  # self.precision
        r = self._safe_divide(self.TP, self.TP + self.FN)  # self.recall
        score = self._safe_divide(2 * p * r, p + r)
        return self.apply_reduction_to_score(score)

    @property
    def specificity(self):
        """特异性是指模型在负类别样本中的识别能力"""
        score = self._safe_divide(self.TN, self.TN + self.FP)
        return self.apply_reduction_to_score(score)

    @property
    def fprate(self):
        """False Positive Rate,假阳率是模型将负类别样本错误分类为正类别的比例"""
        score = self._safe_divide(self.FP, self.FP + self.TN)
        return self.apply_reduction_to_score(score)

    @property
    def fnrate(self):
        """False Negative Rate,假阴率是模型将正类别样本错误分类为负类别的比例"""
        score = self._safe_divide(self.FN, self.FN + self.TP)
        return self.apply_reduction_to_score(score)

    @property
    def qualityfactor(self):
        """品质因子综合考虑了召回率和特异性"""
        r = self._safe_divide(self.TP, self.TP + self.FN)  # self.recall
        s = self._safe_divide(self.TN, self.TN + self.FP)  # self.specificity
        score = r + s - 1
        return self.apply_reduction_to_score(score)

    @property
    def dice(self):
        """Dice系数更多的是用于分割方面"""
        score = self._safe_divide(2 * self.TP, 2 * self.TP + self.FN + self.FP)
        return self.apply_reduction_to_score(score)

    @property
    def kappa(self):
        """Kappa系数，衡量分类器预测结果与真实标签的一致性"""
        total = self.TP + self.TN + self.FP + self.FN
        po = (self.TP + self.TN) / total  # 观察一致性
        pe = ((self.TP + self.FP) * (self.TP + self.FN) + (self.FN + self.TN)*(self.FP + self.TN))/(total ** 2)  # 随机一致性
        score = self._safe_divide(po - pe, 1 - pe)
        return self.apply_reduction_to_score(score)

    @property
    def mcc(self):
        """Matthews Correlation Coefficient，衡量分类器性能的对称性指标"""
        numerator = self.TP * self.TN - self.FP * self.FN
        denominator = np.sqrt((self.TP + self.FP)*(self.TP + self.FN)*(self.TN + self.FP)*(self.TN + self.FN))
        score = self._safe_divide(numerator, denominator)
        return self.apply_reduction_to_score(score)

    @property
    def g_mean(self):
        """G-Mean，综合考虑召回率和特异性"""
        r = self._safe_divide(self.TP, self.TP + self.FN)  # self.recall
        s = self._safe_divide(self.TN, self.TN + self.FP)  # self.specificity
        score = np.sqrt(r * s)
        return self.apply_reduction_to_score(score)

    @property
    def balanced_accuracy(self):
        """Balanced Accuracy，适用于类别不平衡问题"""
        r = self._safe_divide(self.TP, self.TP + self.FN)  # self.recall
        s = self._safe_divide(self.TN, self.TN + self.FP)  # self.specificity
        score = (r + s) / 2
        return self.apply_reduction_to_score(score)

    def get_index(self, keys=None):
        metrics_data={
            'Accuracy': self.accuracy,
            'Precision': self.precision,
            'Recall': self.recall,
            'IOU': self.iou,
            'F1Score': self.f1score,
            'Specificity': self.specificity,
            'FpRate': self.fprate,
            'FnRate': self.fnrate,
            'Qualityfactor': self.qualityfactor,
            'Dice': self.dice,
            'Kappa': self.kappa,
            'MCC': self.mcc,
            'G-Mean': self.g_mean,
            'BalancedAccuracy': self.balanced_accuracy
        }
        if not keys:
            return metrics_data
        result = {}
        for key in keys:
            if key in metrics_data:
                result[key] = metrics_data[key]
            else:
                warnings.warn(f"Invalid metric name: {key}", UserWarning)
                continue

        return result

    def eval(self, metrics_data=None, min_width=13, nums=5):
        metrics_data = self.get_index() if metrics_data == None else metrics_data
        max_key_length = max(len(key) for key in metrics_data.keys())
        total_width = max(min_width, max_key_length + 2)
        centered_width = total_width - 2
        print(f"{self.headline} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print('-' * (total_width + nums + 2))
        for key, value in metrics_data.items():
            print(f"{key:^{centered_width}}: ", end="")
            print(f"{value:.{nums}f}")

        print('-' * (total_width + nums + 2))


if __name__=="__main__":
    confusion_matrix = ConfusionMatrixs(num_classes=3, reduction='none')
    true = torch.tensor([1, 1, 1, 0, 0, 0, 2, 2, 2, 2])
    pred = torch.tensor([1, 0, 0, 0, 2, 1, 0, 0, 2, 2])
    confusion_matrix.update(pred, true)
    # confusion_matrix.plot_confusion_matrix()
    TP, FN, FP, TN = confusion_matrix.ravel()
    print(TP, FN, FP, TN)
    print(confusion_matrix.get_matrix)

    index = ClassificationIndex(TP, FN, FP, TN, reduction='mean')
    index.eval()
