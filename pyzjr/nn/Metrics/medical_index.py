import torch
import torch.nn as nn
import numpy as np
from pyzjr.core.general import is_numpy, is_tensor
from pyzjr.nn.Metrics.indexutils import ignore_background, do_metric_reduction_3d

def get_confusion_matrix_3d_torch(y_pred, y, include_background=True):
    """
    Calculate the three-dimensional confusion matrix of tensor type.
    """
    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    y = y.float()
    y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    batch_size, n_class = y_pred.shape[:2]
    y_pred = y_pred.reshape(batch_size, n_class, -1)
    y = y.reshape(batch_size, n_class, -1)
    tp = ((y_pred + y) == 2).float()
    tn = ((y_pred + y) == 0).float()

    tp = tp.sum(dim=[2])
    tn = tn.sum(dim=[2])
    p = y.sum(dim=[2])
    n = y.shape[-1] - p

    fn = p - tp
    fp = n - tn

    return torch.stack([tp, fp, tn, fn], dim=-1)

def get_confusion_matrix_3d_np(y_pred, y, include_background=True):
    """
    Calculate the three-dimensional confusion matrix of ndarray type.
    """
    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    y = y.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    batch_size, n_class = y_pred.shape[:2]
    y_pred = y_pred.reshape(batch_size, n_class, -1)
    y = y.reshape(batch_size, n_class, -1)
    tp = ((y_pred + y) == 2).astype(np.float32)
    tn = ((y_pred + y) == 0).astype(np.float32)

    tp = np.sum(tp, axis=2)
    tn = np.sum(tn, axis=2)
    p = np.sum(y, axis=2)
    n = y.shape[-1] - p

    fn = p - tp
    fp = n - tn

    return np.stack([tp, fp, tn, fn], axis=-1)

def get_confusion_matrix_3d(y_pred, y, include_background=True):
    """
    Calculate the three-dimensional confusion matrix.

    Args:
    y_pred: input data to compute. It must be one-hot format and first dim is batch.
        The values should be binarized.
    y: ground truth to compute the metric. It must be one-hot format and first dim is batch.
        The values should be binarized.
    include_background: whether to include metric computation on the first channel of
        the predicted output. Defaults to True.
    """
    if is_numpy(y_pred) and is_numpy(y):
        return get_confusion_matrix_3d_np(y_pred, y, include_background)
    if is_tensor(y_pred) and is_tensor(y):
        return get_confusion_matrix_3d_torch(y_pred, y, include_background)

class ConfusionMatrixs3D(nn.Module):
    """
    This function is used to compute confusion matrix related metric.

    Args:
        metric_name: [``"sensitivity"``, ``"specificity"``, ``"precision"``, ``"negative predictive value"``,
        ``"miss rate"``, ``"fall out"``, ``"false discovery rate"``, ``"false omission rate"``,
        ``"prevalence threshold"``, ``"threat score"``, ``"accuracy"``, ``"balanced accuracy"``,
        ``"f1 score"``, ``"matthews correlation coefficient"``, ``"fowlkes mallows index"``,
        ``"informedness"``, ``"markedness"``]

    For more names, you can refer to the "_check_confusion_matrix_metric name" function
    under this category.
    """
    def __init__(self, metric_name, include_background=True, reduction='mean', eps=1e-5):
        super(ConfusionMatrixs3D, self).__init__()
        self.metric = self._check_confusion_matrix_metric_name(metric_name)
        self.include_background = include_background
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y):
        confusion_matrix = get_confusion_matrix_3d_torch(y_pred, y, self.include_background)
        confusion_matrix = self._check_conf_matrix_format(confusion_matrix)
        tp = confusion_matrix[..., 0]
        fp = confusion_matrix[..., 1]
        tn = confusion_matrix[..., 2]
        fn = confusion_matrix[..., 3]
        p = tp + fn
        n = fp + tn
        nan_tensor = torch.tensor(float("nan"), device=confusion_matrix.device)
        if self.metric == "tpr":
            numerator, denominator = tp, p
        elif self.metric == "tnr":
            numerator, denominator = tn, n
        elif self.metric == "ppv":
            numerator, denominator = tp, (tp + fp)
        elif self.metric == "npv":
            numerator, denominator = tn, (tn + fn)
        elif self.metric == "fnr":
            numerator, denominator = fn, p
        elif self.metric == "fpr":
            numerator, denominator = fp, n
        elif self.metric == "fdr":
            numerator, denominator = fp, (fp + tp)
        elif self.metric == "for":
            numerator, denominator = fn, (fn + tn)
        elif self.metric == "pt":
            tpr = torch.where(p > 0, tp / p, nan_tensor)
            tnr = torch.where(n > 0, tn / n, nan_tensor)
            numerator = torch.sqrt(tpr * (1.0 - tnr)) + tnr - 1.0
            denominator = tpr + tnr - 1.0
        elif self.metric == "ts":
            numerator, denominator = tp, (tp + fn + fp)
        elif self.metric == "acc":
            numerator, denominator = (tp + tn), (p + n)
        elif self.metric == "ba":
            tpr = torch.where(p > 0, tp / p, nan_tensor)
            tnr = torch.where(n > 0, tn / n, nan_tensor)
            numerator, denominator = (tpr + tnr), 2.0
        elif self.metric == "f1":
            numerator, denominator = tp * 2.0, (tp * 2.0 + fn + fp)
        elif self.metric == "mcc":
            numerator = tp * tn - fp * fn
            denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        elif self.metric == "fm":
            tpr = torch.where(p > 0, tp / p, nan_tensor)
            ppv = torch.where((tp + fp) > 0, tp / (tp + fp), nan_tensor)
            numerator = torch.sqrt(ppv * tpr)
            denominator = 1.0
        elif self.metric == "bm":
            tpr = torch.where(p > 0, tp / p, nan_tensor)
            tnr = torch.where(n > 0, tn / n, nan_tensor)
            numerator = tpr + tnr - 1.0
            denominator = 1.0
        elif self.metric == "mk":
            ppv = torch.where((tp + fp) > 0, tp / (tp + fp), nan_tensor)
            npv = torch.where((tn + fn) > 0, tn / (tn + fn), nan_tensor)
            numerator = ppv + npv - 1.0
            denominator = 1.0
        else:
            raise NotImplementedError("the metric is not implemented.")
        index = numerator / (denominator + self.eps)

        f, not_nans = do_metric_reduction_3d(index, self.reduction)  # type: ignore
        return f

    def _check_conf_matrix_format(self, confusion_matrix):
        input_dim = confusion_matrix.ndimension()
        if input_dim == 1:
            confusion_matrix = confusion_matrix.unsqueeze(dim=0)
        if confusion_matrix.shape[-1] != 4:
            raise ValueError("the size of the last dimension of confusion_matrix should be 4.")
        return confusion_matrix

    def _check_confusion_matrix_metric_name(self, metric_name):
        """
        This function is used to check and simplify compliant names.
        There are many metrics related to confusion matrices, some of which have more than
        one name. This function is used to check and simplify names.
        """
        metric_name = metric_name.replace(" ", "_")
        metric_name = metric_name.lower()
        if metric_name in ["sensitivity", "recall", "hit_rate", "true_positive_rate", "tpr"]:
            return "tpr"
        if metric_name in ["specificity", "selectivity", "true_negative_rate", "tnr"]:
            return "tnr"
        if metric_name in ["precision", "positive_predictive_value", "ppv"]:
            return "ppv"
        if metric_name in ["negative_predictive_value", "npv"]:
            return "npv"
        if metric_name in ["miss_rate", "false_negative_rate", "fnr"]:
            return "fnr"
        if metric_name in ["fall_out", "false_positive_rate", "fpr"]:
            return "fpr"
        if metric_name in ["false_discovery_rate", "fdr"]:
            return "fdr"
        if metric_name in ["false_omission_rate", "for"]:
            return "for"
        if metric_name in ["prevalence_threshold", "pt"]:
            return "pt"
        if metric_name in ["threat_score", "critical_success_index", "ts", "csi"]:
            return "ts"
        if metric_name in ["accuracy", "acc"]:
            return "acc"
        if metric_name in ["balanced_accuracy", "ba"]:
            return "ba"
        if metric_name in ["f1_score", "f1"]:
            return "f1"
        if metric_name in ["matthews_correlation_coefficient", "mcc"]:
            return "mcc"
        if metric_name in ["fowlkes_mallows_index", "fm"]:
            return "fm"
        if metric_name in ["informedness", "bookmaker_informedness", "bm", "youden_index", "youden"]:
            return "bm"
        if metric_name in ["markedness", "deltap", "mk"]:
            return "mk"
        raise NotImplementedError("the metric is not implemented.")

if __name__=="__main__":
    metric_name = "acc"

    confusion_metric = ConfusionMatrixs3D(metric_name)
    batch_size = 2
    num_channels = 3
    depth = 2
    height = 2
    width = 2
    y_pred = torch.randint(0, num_channels, (batch_size, num_channels, depth, height, width))
    y_true = torch.randint(0, num_channels, (batch_size, num_channels, depth, height, width))
    result = confusion_metric(y_pred, y_true)

    print(f"{metric_name}: {result}")