"""
Copyright (c) 2024, Auorui.
All rights reserved.
Indicator detection for semantic segmentation, only used for multilabel.
Referring to https://smp.readthedocs.io/en/latest/metrics.html
                 | Predicted Positive | Predicted Negative |
Actual Positive  |        TP          |         FN         |
Actual Negative  |        FP          |         TN         |
Time: 2024-01-26
"""
import torch
import warnings
from datetime import datetime

@torch.no_grad()
def calculate_confusion_matrix_multilabel(output, target, threshold=None):
    if torch.is_floating_point(target):
        raise ValueError(f"Target should be one of the integer types, got {target.dtype}.")

    if torch.is_floating_point(output) and threshold is None:
        raise ValueError(
            f"Output should be one of the integer types if ``threshold`` is not None, got {output.dtype}."
        )
    if output.shape != target.shape:
        raise ValueError(
            "Dimensions should match, but ``output`` shape is not equal to ``target`` "
            + f"shape, {output.shape} != {target.shape}"
        )
    if threshold is not None:
        output = torch.where(output >= threshold, 1, 0)
        target = torch.where(target >= threshold, 1, 0)
    batch_size, num_classes, *dims = target.shape
    output = output.view(batch_size, num_classes, -1)
    target = target.view(batch_size, num_classes, -1)

    # TP, FP, FN, TN
    tp = (output * target).sum(2)
    fp = output.sum(2) - tp
    fn = target.sum(2) - tp
    tn = torch.prod(torch.tensor(dims)) - (tp + fp + fn)

    return tp, fn, fp, tn

def _fbeta_score(tp, fn, fp, tn, beta=1):
    beta_tp = (1 + beta**2) * tp
    beta_fn = (beta**2) * fn
    score = beta_tp / (beta_tp + beta_fn + fp)
    return score

def _iou_score(tp, fn, fp, tn):
    return tp / (tp + fp + fn)


def _accuracy(tp, fn, fp, tn):
    return (tp + tn) / (tp + fp + fn + tn)


def _dice_coefficient(tp, fn, fp, tn):
    return (2 * tp) / (2 * tp + fp + fn)


def _recall(tp, fn, fp, tn):
    return tp / (tp + fn)


def _precision(tp, fn, fp, tn):
    return tp / (tp + fp)


def _sensitivity(tp, fn, fp, tn):
    return tp / (tp + fn)


def _specificity(tp, fn, fp, tn):
    return tn / (tn + fp)


def _balanced_accuracy(tp, fn, fp, tn):
    return (_sensitivity(tp, fn, fp, tn) + _specificity(tp, fn, fp, tn)) / 2


def _positive_predictive_value(tp, fn, fp, tn):
    return tp / (tp + fp)


def _negative_predictive_value(tp, fn, fp, tn):
    return tn / (tn + fn)


def _false_negative_rate(tp, fn, fp, tn):
    return fn / (fn + tp)


def _false_positive_rate(tp, fn, fp, tn):
    return fp / (fp + tn)


def _false_discovery_rate(tp, fn, fp, tn):
    return 1 - _positive_predictive_value(tp, fn, fp, tn)


def _false_omission_rate(tp, fn, fp, tn):
    return 1 - _negative_predictive_value(tp, fn, fp, tn)


def _positive_likelihood_ratio(tp, fn, fp, tn):
    return _sensitivity(tp, fn, fp, tn) / _false_positive_rate(tp, fn, fp, tn)


def _negative_likelihood_ratio(tp, fn, fp, tn):
    return _false_negative_rate(tp, fn, fp, tn) / _specificity(tp, fn, fp, tn)

class SegmentationIndex():
    """
    Segmentation metrics calculation class for multi-label segmentation tasks.
    Supports calculation of various segmentation evaluation metrics (such as F1 score, IOU, accuracy, recall, precision, etc.)
    with options to aggregate the results in different ways (e.g., micro, macro, weighted, etc.).

    Parameters:
        tp (Tensor): True positives, a tensor of shape (N, C), where N is the number of samples and C is the number of classes.
        fn (Tensor): False negatives, a tensor of the same shape as tp.
        fp (Tensor): False positives, a tensor of the same shape as tp.
        tn (Tensor): True negatives, a tensor of the same shape as tp.
        reduction (str, optional): Aggregation method. Possible values are "micro", "macro", "weighted", "micro-imagewise",
                                   "macro-imagewise", "weighted-imagewise", "none", or None. Default is 'mean'.
        class_weights (Tensor, optional): Weights for each class used in weighted averaging. Defaults to None, which applies equal weights.
        zero_division (float, optional): Handling of division by zero. Defaults to 1.0, which returns 1.0 in case of zero division.
                                        Another option is "warn", which issues a warning when division by zero occurs.
        headline (str, optional): Title to be displayed in the output. Default is "SegmentationIndex".

    Methods:
        handle_zero_division(x): Handles division by zero, returning the appropriate value based on zero_division.
        apply_reduction_to_score(metricfn, **kwargs): Applies the specified metric function and computes the final score
                                                     based on the selected reduction method.
        fbeta_score(beta): Computes the Fβ score, where β is a tunable parameter that balances precision and recall.
        iou_score: Computes the Intersection Over Union (IOU) score.
        accuracy: Computes the accuracy score.
        recall: Computes the recall score.
        precision: Computes the precision score.
        sensitivity: Computes the sensitivity score.
        specificity: Computes the specificity score.
        dice_coefficient: Computes the dice coefficient.
        balanced_accuracy: Computes the balanced accuracy score.
        positive_predictive_value: Computes the Positive Predictive Value (PPV).
        negative_predictive_value: Computes the Negative Predictive Value (NPV).
        false_negative_rate: Computes the False Negative Rate (FNR).
        false_positive_rate: Computes the False Positive Rate (FPR).
        false_discovery_rate: Computes the False Discovery Rate (FDR).
        false_omission_rate: Computes the False Omission Rate (FOR).
        positive_likelihood_ratio: Computes the Positive Likelihood Ratio (PLR).
        negative_likelihood_ratio: Computes the Negative Likelihood Ratio (NLR).
        get_index(keys=None): Retrieves the values for the specified metrics. Returns all metrics if no keys are specified.
        compute(metrics_data=None, min_width=13, nums=5): Prints the computed results, allowing for sorting by metric name,
                                                        and controlling the output width and number of decimal places.
    """
    def __init__(self, tp, fn, fp, tn, reduction='mean', class_weights=None,
                 zero_division=1.0, headline="SegmentationIndex"):
        self.reduction = reduction
        self.class_weights = class_weights
        self.zero_division = zero_division
        self.headline = headline
        self.tp, self.fn, self.fp, self.tn = tp, fn, fp, tn

    def handle_zero_division(self, x):
        nans = torch.isnan(x)
        if torch.any(nans) and self.zero_division == "warn":
            warnings.warn("Zero division in metric calculation!")
        value = self.zero_division if self.zero_division != "warn" else 0
        value = torch.tensor(value, dtype=x.dtype).to(x.device)
        x = torch.where(nans, value, x)
        return x

    def apply_reduction_to_score(self, metricfn, **kwargs):
        if self.class_weights is None and self.reduction == "weighted":
            raise ValueError(f"Class weights should be provided for `{self.reduction}` reduction")
        class_weights = self.class_weights if self.class_weights is not None else 1.0
        class_weights = torch.tensor(class_weights).to(self.tp.device)
        class_weights = class_weights / class_weights.sum()
        if self.reduction == "micro":
            tp = self.tp.sum()
            fp = self.fp.sum()
            fn = self.fn.sum()
            tn = self.tn.sum()
            score = metricfn(tp, fn, fp, tn, **kwargs)
        elif self.reduction == "macro":
            tp = self.tp.sum(0)
            fp = self.fp.sum(0)
            fn = self.fn.sum(0)
            tn = self.tn.sum(0)
            score = metricfn(tp, fn, fp, tn, **kwargs)
            score = self.handle_zero_division(score)
            score = (score * class_weights).mean()
        elif self.reduction == "weighted":
            tp = self.tp.sum(0)
            fp = self.fp.sum(0)
            fn = self.fn.sum(0)
            tn = self.tn.sum(0)
            score = metricfn(tp, fn, fp, tn, **kwargs)
            score = self.handle_zero_division(score)
            score = (score * class_weights).sum()
        elif self.reduction == "micro-imagewise":
            tp = self.tp.sum(1)
            fp = self.fp.sum(1)
            fn = self.fn.sum(1)
            tn = self.tn.sum(1)
            score = metricfn(tp, fn, fp, tn, **kwargs)
            score = self.handle_zero_division(score)
            score = score.mean()
        elif self.reduction == "macro-imagewise" or self.reduction == "weighted-imagewise":
            score = metricfn(self.tp, self.fn, self.fp, self.tn, **kwargs)
            score = self.handle_zero_division(score)
            score = (score.mean(0) * class_weights).mean()
        elif self.reduction == "none" or self.reduction is None:
            score = metricfn(self.tp, self.fn, self.fp, self.tn, **kwargs)
            score = self.handle_zero_division(score)
        else:
            raise ValueError(
                "`reduction` should be in "
                "[micro, macro, weighted, micro-imagewise, macro-imagesize, weighted-imagewise, none, None]"
            )
        return score

    def fbeta_score(self, beta):
        return self.apply_reduction_to_score(
            _fbeta_score, beta=beta
        )

    @property
    def f1_score(self):
        return self.fbeta_score(beta=1.0)

    @property
    def iou_score(self):
        return self.apply_reduction_to_score(_iou_score)

    @property
    def accuracy(self):
        return self.apply_reduction_to_score(_accuracy)

    @property
    def recall(self):
        return self.apply_reduction_to_score(_recall)

    @property
    def precision(self):
        return self.apply_reduction_to_score(_precision)

    @property
    def sensitivity(self):
        return self.apply_reduction_to_score(_sensitivity)

    @property
    def specificity(self):
        return self.apply_reduction_to_score(_specificity)

    @property
    def dice_coefficient(self):
        return self.apply_reduction_to_score(_dice_coefficient)

    @property
    def balanced_accuracy(self):
        return self.apply_reduction_to_score(_balanced_accuracy)

    @property
    def positive_predictive_value(self):
        return self.apply_reduction_to_score(_positive_predictive_value)

    @property
    def negative_predictive_value(self):
        return self.apply_reduction_to_score(_negative_predictive_value)

    @property
    def false_negative_rate(self):
        return self.apply_reduction_to_score(_false_negative_rate)

    @property
    def false_positive_rate(self):
        return self.apply_reduction_to_score(_false_positive_rate)

    @property
    def false_discovery_rate(self):
        return self.apply_reduction_to_score(_false_discovery_rate)

    @property
    def false_omission_rate(self):
        return self.apply_reduction_to_score(_false_omission_rate)

    @property
    def positive_likelihood_ratio(self):
        return self.apply_reduction_to_score(_positive_likelihood_ratio)

    @property
    def negative_likelihood_ratio(self):
        return self.apply_reduction_to_score(_negative_likelihood_ratio)

    def get_index(self, keys=None):
        metrics_data = {
            'f1_score': self.f1_score,
            'iou_score': self.iou_score,
            'accuracy': self.accuracy,
            'recall': self.recall,
            'precision': self.precision,
            'sensitivity': self.sensitivity,
            'specificity': self.specificity,
            'dice_coefficient': self.dice_coefficient,
            'balanced_accuracy': self.balanced_accuracy,
            'positive_predictive_value': self.positive_predictive_value,
            'negative_predictive_value': self.negative_predictive_value,
            'false_negative_rate': self.false_negative_rate,
            'false_positive_rate': self.false_positive_rate,
            'false_discovery_rate': self.false_discovery_rate,
            'false_omission_rate': self.false_omission_rate,
            'positive_likelihood_ratio': self.positive_likelihood_ratio,
            'negative_likelihood_ratio': self.negative_likelihood_ratio,
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
        metrics_data = self.get_index() if metrics_data is None else metrics_data
        max_key_length = max(len(key) for key in metrics_data.keys())
        total_width = max(min_width, max_key_length + 2)
        centered_width = total_width - 2
        print(f"{self.headline} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print('-' * (total_width + nums + 2))
        for key, value in metrics_data.items():
            print(f"{key:^{centered_width}}: ", end="")
            print(f"{value:.{nums}f}")

        print('-' * (total_width + nums + 2))


def dice_coefficient(pred, true, reduction='micro', esp=1e-5):
    """
    Calculate the Dice coefficient for multi label segmentation tasks, supporting different
    reduction methods.
    :param pred: predictive value
    :param true: Label, True value
    :param reduction: micro average:Global Dice.All pixels (regardless of their category) are equally important
            macro average:Dice by category.All categories (regardless of the number of pixels) are equally important
            'none' or None: the Dice coefficient for each category
    :param esp: Smooth coefficient, preventing division by 0
    """
    dice_score = 0
    if pred.shape != true.shape:
        raise ValueError(f"predicted value and true value "
                         f"should have same shape, but got predicted value shape: {pred.shape}, "
                         f"true value shape: {true.shape}.")
    if reduction == 'micro':
        intersection = torch.sum(pred * true)
        denominator = torch.sum(pred) + torch.sum(true)
        dice_score = (2.0 * intersection) / (denominator + esp)
    elif reduction == 'macro':
        intersection = torch.sum(pred * true, dim=(0, 2, 3))
        denominator = torch.sum(pred, dim=(0, 2, 3)) + torch.sum(true, dim=(0, 2, 3))
        dice_score = (2.0 * intersection) / (denominator + esp)
        dice_score = dice_score.mean()
    elif reduction in ['none', None]:
        intersection = torch.sum(pred * true, dim=(0, 2, 3))
        denominator = torch.sum(pred, dim=(0, 2, 3)) + torch.sum(true, dim=(0, 2, 3))
        dice_score = (2.0 * intersection) / (denominator + esp)
    return dice_score

def iou_score(pred, true, reduction='micro', esp=1e-5):
    """
    Calculate the iou for multi label segmentation tasks, supporting different
    reduction methods.
    :param pred: predictive value
    :param true: Label, True value
    :param reduction: micro average:Global iou score.All pixels (regardless of their category) are equally important
            macro average:Dice by category.All categories (regardless of the number of pixels) are equally important
            'none' or None: the iou score for each category
    :param esp: Smooth coefficient, preventing division by 0
    """
    iou_scores = 0
    if pred.shape != true.shape:
        raise ValueError(f"predicted value and true value "
                         f"should have same shape, but got predicted value shape: {pred.shape}, "
                         f"true value shape: {true.shape}.")
    if reduction == 'micro':
        intersection = torch.sum(pred * true)
        union = torch.sum(pred) + torch.sum(true) - intersection
        iou_scores = intersection / (union + esp)

    elif reduction == 'macro':
        intersection = torch.sum(pred * true, dim=(0, 2, 3))
        sum_pred = torch.sum(pred, dim=(0, 2, 3))
        sum_true = torch.sum(true, dim=(0, 2, 3))
        union = sum_pred + sum_true - intersection
        iou_per_class = intersection / (union + esp)
        iou_scores = torch.mean(iou_per_class)

    elif reduction in ['none', None]:
        intersection = torch.sum(pred * true, dim=(0, 2, 3))
        sum_pred = torch.sum(pred, dim=(0, 2, 3))
        sum_true = torch.sum(true, dim=(0, 2, 3))
        union = sum_pred + sum_true - intersection
        iou_scores = intersection / (union + esp)
    return iou_scores

if __name__=="__main__":
    from pyzjr.nn.metrics.utils import generate_class_weights
    output = torch.rand([2, 4, 10, 10])
    target = torch.rand([2, 4, 10, 10]).round().long()

    class_weights = generate_class_weights(target, num_classes=4, strategy='inverse_frequency')
    tp, fn, fp, tn = calculate_confusion_matrix_multilabel(output, target, .5)
    print(tp, fn, fp, tn)

    index = SegmentationIndex(tp, fn, fp, tn, reduction='micro', class_weights=None)
    metrics_data = index.get_index(['f1_score', 'negative_likelihood_ratio',
                                    'balanced_accuracy', 'recall'])
    print(metrics_data)
    index.eval(metrics_data)
    index.eval()