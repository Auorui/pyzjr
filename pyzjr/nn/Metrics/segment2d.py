"""
Copyright (c) 2024, Auorui.
All rights reserved.
用于 语义分割 的指标检测
                 | Predicted Positive | Predicted Negative |
Actual Positive  |        TP          |         FN         |
Actual Negative  |        FP          |         TN         |
Time: 2024-01-26
"""
import torch
import numpy as np


__all__ = ["Miou",
           "Recall",
           "Precision",
           "F1Score",
           "DiceCoefficient",
           "Accuracy",
           "SegmentationIndex",
           "AIU"]

class Miou(object):
    """
    iou = ( A & B ) / ( A | B )
    """
    def __init__(self, num_classes, ignore=None, threshold=.5, esp=1e-5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.esp = esp
        self.ignore = ignore

    def __call__(self, pred, true):
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(true):
            true = true.cpu().numpy()
        if self.ignore:
            pred = pred[:, :self.ignore, :, :]
            true = true[:, :self.ignore, :, :]

        assert pred.shape[1] == true.shape[1] == self.num_classes, "Number of classes mismatch"
        pred = pred > self.threshold
        true = true > self.threshold
        intersection = (pred & true).sum(axis=(1, 2, 3))
        union = (pred | true).sum(axis=(1, 2, 3))

        iou = intersection / (union + self.esp)
        return iou.mean()

class Recall(object):
    """
    recall = TP / (TP + FN)
    """
    def __init__(self, num_classes, ignore=None, threshold=.5, esp=1e-5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.esp = esp
        self.ignore = ignore

    def __call__(self, pred, true):
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(true):
            true = true.cpu().numpy()
        if self.ignore:
            pred = pred[:, :self.ignore, :, :]
            true = true[:, :self.ignore, :, :]

        assert pred.shape[1] == true.shape[1] == self.num_classes, "Number of classes mismatch"
        pred = pred > self.threshold
        true = true > self.threshold

        TP = (pred & true).sum(axis=(1, 2, 3))
        FN = (~pred & true).sum(axis=(1, 2, 3))

        recall = TP / (TP + FN + self.esp)
        return recall.mean()

class Precision(object):
    """
    precision = TP / (TP + FP)
    """
    def __init__(self, num_classes, ignore=None, threshold=.5, esp=1e-5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.esp = esp
        self.ignore = ignore

    def __call__(self, pred, true):
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(true):
            true = true.cpu().numpy()
        if self.ignore:
            pred = pred[:, :self.ignore, :, :]
            true = true[:, :self.ignore, :, :]

        assert pred.shape[1] == true.shape[1] == self.num_classes, "Number of classes mismatch"
        pred = pred > self.threshold
        true = true > self.threshold

        TP = (pred & true).sum(axis=(1, 2, 3))
        FP = (pred & ~true).sum(axis=(1, 2, 3))

        precision = TP / (TP + FP + self.esp)
        return precision.mean()

class Accuracy(object):
    """
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    def __init__(self, num_classes, ignore=None, threshold=.5, esp=1e-5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.esp = esp
        self.ignore = ignore

    def __call__(self, pred, true):
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(true):
            true = true.cpu().numpy()
        if self.ignore:
            pred = pred[:, :self.ignore, :, :]
            true = true[:, :self.ignore, :, :]
        assert pred.shape[1] == true.shape[1] == self.num_classes, "Number of classes mismatch"
        pred = pred > self.threshold
        true = true > self.threshold

        TP = (pred & true).sum(axis=(1, 2, 3))
        TN = (~pred & ~true).sum(axis=(1, 2, 3))
        FP = (pred & ~true).sum(axis=(1, 2, 3))
        FN = (~pred & true).sum(axis=(1, 2, 3))

        accuracy = (TP + TN) / (TP + TN + FP + FN + self.esp)
        return accuracy.mean()

class F1Score(object):
    """
    f1score = 2 * (precision * recall) / (precision + recall)
    """
    def __init__(self, num_classes, ignore=None, threshold=.5, esp=1e-5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.esp = esp
        self.ignore = ignore

    def __call__(self, pred, true):
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(true):
            true = true.cpu().numpy()
        if self.ignore:
            pred = pred[:, :self.ignore, :, :]
            true = true[:, :self.ignore, :, :]
        pred = pred > self.threshold
        true = true > self.threshold

        recall = Recall(self.num_classes)(pred, true)
        precision = Precision(self.num_classes)(pred, true)
        f1_score = 2 * (precision * recall) / (precision + recall + self.esp)
        return f1_score.mean()

class DiceCoefficient(object):
    """
    dice_coefficient = (2 * TP) / (2 * TP + FP + FN)
    """
    def __init__(self, num_classes, ignore=None, threshold=.5, esp=1e-5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.esp = esp
        self.ignore = ignore

    def __call__(self, pred, true):
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(true):
            true = true.cpu().numpy()
        if self.ignore:
            pred = pred[:, :self.ignore, :, :]
            true = true[:, :self.ignore, :, :]
        assert pred.shape[1] == true.shape[1] == self.num_classes, "Number of classes mismatch"
        pred = pred > self.threshold
        true = true > self.threshold

        TP = (pred & true).sum(axis=(1, 2, 3))
        FP = (pred & ~true).sum(axis=(1, 2, 3))
        FN = (~pred & true).sum(axis=(1, 2, 3))

        dice_coefficient = (2 * TP) / (2 * TP + FP + FN + self.esp)
        return dice_coefficient.mean()

class SegmentationIndex(object):
    def __init__(self, num_classes, ignore=None, headline="Semantic Segmentation Index:",threshold=.5, esp=1e-5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.ignore = ignore
        self.esp = esp
        self.miou = Miou(self.num_classes, self.ignore)
        self.recall = Recall(self.num_classes, self.ignore)
        self.precision = Precision(self.num_classes, self.ignore)
        self.f1Score = F1Score(self.num_classes, self.ignore)
        self.dice_coefficient = DiceCoefficient(self.num_classes, self.ignore)
        self.acc = Accuracy(self.num_classes, self.ignore)
        self.index = []
        self.headline = headline

    def update(self, pred, true):
        pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
        true = true.cpu().numpy() if torch.is_tensor(true) else true
        miou_value = self.miou(pred, true)
        recall_value = self.recall(pred, true)
        precision_value = self.precision(pred, true)
        f1score_value = self.f1Score(pred, true)
        dice_value = self.dice_coefficient(pred, true)
        acc_value = self.acc(pred, true)

        self.index.append({
            'miou': miou_value,
            'recall': recall_value,
            'precision': precision_value,
            'f1score': f1score_value,
            'dice': dice_value,
            'accuracy': acc_value
        })
        return miou_value, recall_value, precision_value, f1score_value, dice_value, acc_value

    def __str__(self):
        latest_index = self.index[-1] if self.index else None

        return (f"\033[94m{self.headline}\n"
                f"\t  Miou         Recall         Precision         F1Score         "
                f"DiceCoefficient         Accuracy\n\033[94m"
                f"\t{latest_index['miou']:.5f}       {latest_index['recall']:.5f}          "
                f"{latest_index['precision']:.5f}          {latest_index['f1score']:.5f}            "
                f"{latest_index['dice']:.5f}              {latest_index['accuracy']:.5f}\033[0m")


class AIU(object):
    """
    Reference from paper https://arxiv.org/pdf/1901.06340.pdf Formula 7:
    aiu = [Nt_pg / (Nt_p + Nt_g - Nt_pg] / N
    """
    def __init__(self, num_classes, ignore=None, threshold=(1, 100), esp=1e-5):
        self.num_classes = num_classes
        self.thresholds = [i / 100.0 for i in range(*threshold)]  # 从0.01到0.99，间隔0.01
        self.esp = esp
        self.ignore = ignore

    def __call__(self, pred, true):
        total_aiu = 0.
        pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
        true = true.cpu().numpy() if torch.is_tensor(true) else true
        if self.ignore:
            pred = pred[:, :self.ignore, :, :]
            true = true[:, :self.ignore, :, :]

        assert pred.shape[1] == true.shape[1] == self.num_classes, "Number of classes mismatch"
        for threshold in self.thresholds:
            intersection = (pred > threshold) & (true > threshold)
            Nt_pg = intersection.sum(axis=(1, 2, 3))
            Nt_p = (pred > threshold).sum(axis=(1, 2, 3))
            Nt_g = (true > threshold).sum(axis=(1, 2, 3))

            aiu_t = Nt_pg / (Nt_p + Nt_g - Nt_pg + self.esp)  # 添加小的常数以避免除零错误
            total_aiu += aiu_t
        average_aiu = total_aiu / len(self.thresholds)
        return np.array(average_aiu).mean()

if __name__ == "__main__":
    batch_size, num_classes, height, width = 2, 4, 3, 3

    pred = torch.randn((batch_size, num_classes, height, width))
    true = torch.randint(0, 2, (batch_size, num_classes, height, width))

    print(pred.shape, true.shape)
    # pred = torch.Tensor([[[[-0.1670,  0.1940,  0.4754],
    #                        [-0.0729,  1.1101, -0.4133],
    #                        [ 0.1728,  0.2493, -0.8165]],
    #
    #                       [[ 0.7978,  0.7136, -0.9680],
    #                        [ 0.1166,  0.2708,  1.5798],
    #                        [ 0.5725,  0.5781, -0.3959]]]])
    # print(pred.shape)
    # true = torch.Tensor([[[[0, 0, 0],
    #                        [0, 0, 1],
    #                        [1, 1, 0]],
    #
    #                       [[1, 1, 1],
    #                        [0, 0, 1],
    #                        [1, 1, 0]],
    #
    #                       # [[0, 0, 0],
    #                       #  [1, 0, 1],
    #                       #  [0, 1, 1]]
    #                       ]])
    # print(true.shape)
    ignore = None
    miou = Miou(num_classes, ignore=ignore)(pred, true)
    recall = Recall(num_classes, ignore=ignore)(pred, true)
    precision = Precision(num_classes, ignore=ignore)(pred, true)
    accuracy = Accuracy(num_classes, ignore=ignore)(pred, true)
    f1score = F1Score(num_classes, ignore=ignore)(pred, true)
    dice = DiceCoefficient(num_classes, ignore=ignore)(pred, true)
    aiu = AIU(num_classes, ignore=ignore)(pred, true)

    print("Mean IoU:", miou)
    print("Recall:", recall)
    print("Precision:", precision)
    print("Accuracy:", accuracy)
    print("F1Score:", f1score)
    print("DiceCoefficient:", dice)
    print("AIU:", aiu)

    # segmentation_index = SegmentationIndex(num_classes, ignore=ignore)
    # segmentation_index.update(pred, true)
    # print(segmentation_index)