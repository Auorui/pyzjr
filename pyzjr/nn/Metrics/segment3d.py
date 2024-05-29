import torch
import torch.nn as nn
import numpy as np
from pyzjr.nn.Metrics.indexutils import do_metric_reduction_3d, hd, ignore_background
from pyzjr.core.general import is_tensor

class DiceMetric3d(nn.Module):
    def __init__(
            self,
            include_background=True,
            reduction='mean',
            get_not_nans=False,
            ignore_empty=True,
            num_classes=None,
            softmax=False,
            sigmoid=False
    ):
        super(DiceMetric3d, self).__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.ignore_empty = ignore_empty
        self.num_classes = num_classes
        self.softmax = softmax
        self.sigmoid = sigmoid

    def compute_channel(self, y_pred, y):
        y_o = torch.sum(y)
        if y_o > 0:
            return (2.0 * torch.sum(torch.masked_select(y, y_pred))) / (y_o + torch.sum(y_pred))
        if self.ignore_empty:
            return torch.tensor(float("nan"), device=y_o.device)
        denorm = y_o + torch.sum(y_pred)
        if denorm <= 0:
            return torch.tensor(1.0, device=y_o.device)
        return torch.tensor(0.0, device=y_o.device)

    def forward(self, y_pred, y):
        """
        Args:
            y_pred: input predictions with shape (batch_size, num_classes or 1, spatial_dims...).
                the number of channels is inferred from ``y_pred.shape[1]`` when ``num_classes is None``.
            y: ground truth with shape (batch_size, num_classes or 1, spatial_dims...).
        """
        _softmax, _sigmoid = self.softmax, self.sigmoid
        if self.num_classes is None:
            n_pred_ch = y_pred.shape[1]  # y_pred is in one-hot format or multi-channel scores
        else:
            n_pred_ch = self.num_classes
            if y_pred.shape[1] == 1 and self.num_classes > 1:  # y_pred is single-channel class indices
                _softmax = _sigmoid = False

        if _softmax:
            if n_pred_ch > 1:
                y_pred = torch.argmax(y_pred, dim=1, keepdim=True)

        elif _sigmoid:
            if self.activate:
                y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred > 0.5

        first_ch = 0 if self.include_background else 1
        data = []
        for b in range(y_pred.shape[0]):
            c_list = []
            for c in range(first_ch, n_pred_ch) if n_pred_ch > 1 else [1]:
                x_pred = (y_pred[b, 0] == c) if (y_pred.shape[1] == 1) else y_pred[b, c].bool()
                x = (y[b, 0] == c) if (y.shape[1] == 1) else y[b, c]
                c_list.append(self.compute_channel(x_pred, x))
            data.append(torch.stack(c_list))
        data = torch.stack(data, dim=0).contiguous()

        f, not_nans = do_metric_reduction_3d(data, self.reduction)  # type: ignore
        return (f, not_nans) if self.get_not_nans else f

class HausdorffDistanceMetric3d(nn.Module):
    def __init__(
            self,
            include_background=True,
            percentile=None,
    ):
        super(HausdorffDistanceMetric3d, self).__init__()
        self.include_background = include_background
        self.percentile = percentile

    def forward(self, y_pred, y):
        if not self.include_background:
            y_pred, y = ignore_background(y_pred=y_pred, y=y)
        if y.shape != y_pred.shape:
            raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

        batch_size, n_class = y_pred.shape[:2]
        hd_ = []
        if is_tensor(y_pred) and is_tensor(y):
            y_pred, y = y_pred.numpy(), y.numpy()

        for c in np.ndindex(n_class):
            surface_distance = hd(y_pred[:, c,...], y[:, c,...])
            hd_.append(surface_distance)

        if 0 <= self.percentile <= 100:
            return np.percentile(hd_, self.percentile)

class MeanIoUMetric3d(nn.Module):
    def __init__(
            self,
            include_background=True,
            reduction='mean',
            get_not_nans=False,
            ignore_empty=True,
    ):
        super().__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.ignore_empty = ignore_empty

    def forward(self, y_pred, y):
        if not self.include_background:
            y_pred, y = ignore_background(y_pred=y_pred, y=y)

        y = y.float()
        y_pred = y_pred.float()

        if y.shape != y_pred.shape:
            raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

        # reducing only spatial dimensions (not batch nor channels)
        n_len = len(y_pred.shape)
        reduce_axis = list(range(2, n_len))
        intersection = torch.sum(y * y_pred, dim=reduce_axis)

        y_o = torch.sum(y, reduce_axis)
        y_pred_o = torch.sum(y_pred, dim=reduce_axis)
        union = y_o + y_pred_o - intersection

        if self.ignore_empty:
            dice = torch.where(y_o > 0, (intersection) / union, torch.tensor(float("nan"), device=y_o.device))
        else:
            dice = torch.where(union > 0, (intersection) / union, torch.tensor(1.0, device=y_o.device))

        f, not_nans = do_metric_reduction_3d(dice, self.reduction)  # type: ignore
        return (f, not_nans) if self.get_not_nans else f

if __name__=="__main__":
    batch_size = 1
    num_classes = 3
    depth = 5
    height = 10
    width = 10
    y_pred = torch.rand(batch_size, num_classes, depth, height, width)
    y = torch.randint(0, num_classes, (batch_size, num_classes, depth, height, width))
    dice_metric = DiceMetric3d(include_background=True, reduction="none")
    hd_metric = HausdorffDistanceMetric3d(include_background=True, percentile=95)
    iou_metric = MeanIoUMetric3d(include_background=True, reduction="none")
    dice_score = dice_metric(y_pred, y)
    print("Dice Score:", dice_score)
    hd_score = hd_metric(y_pred, y)
    print("HD Score:", hd_score)
    iou_score = iou_metric(y_pred, y)
    print("IOU Score:", iou_score)