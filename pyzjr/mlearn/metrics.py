# import torch
# from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
#
# def roc_auc_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):
#     y_true = y_targets.cpu().numpy()
#     y_pred = y_preds.cpu().numpy()
#     return roc_auc_score(y_true, y_pred)
#
#
# def roc_auc_curve_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):
#     y_true = y_targets.cpu().numpy()
#     y_pred = y_preds.cpu().numpy()
#     return roc_curve(y_true, y_pred)
#
# def precision_recall_curve_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):
#     try:
#         from sklearn.metrics import precision_recall_curve
#     except ImportError:
#         raise ModuleNotFoundError("This contrib module requires scikit-learn to be installed.")
#
#     y_true = y_targets.cpu().numpy()
#     y_pred = y_preds.cpu().numpy()
#     return precision_recall_curve(y_true, y_pred)
#
# def average_precision_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
#     y_true = y_targets.cpu().numpy()
#     y_pred = y_preds.cpu().numpy()
#     return average_precision_score(y_true, y_pred)
#
#
# class MultiLabelConfusionMatrix():
#     """Calculates a confusion matrix for multi-labelled, multi-class data."""
#     def __init__(self,num_classes,output_transform,device=torch.device("cpu"),normalized = False,):
#         if num_classes <= 1:
#             raise ValueError("Argument num_classes needs to be > 1")
#         self.num_classes = num_classes
#         self.normalized = normalized
#
#     def reset(self):
#         self.confusion_matrix = torch.zeros(self.num_classes, 2, 2, dtype=torch.int64, device=self._device)
#
#     def update(self, output):
#         self._check_input(output)
#         y_pred, y = output[0].detach(), output[1].detach()
#
#         y_reshaped = y.transpose(0, 1).reshape(self.num_classes, -1)
#         y_pred_reshaped = y_pred.transpose(0, 1).reshape(self.num_classes, -1)
#
#         y_total = y_reshaped.sum(dim=1)
#         y_pred_total = y_pred_reshaped.sum(dim=1)
#
#         tp = (y_reshaped * y_pred_reshaped).sum(dim=1)
#         fp = y_pred_total - tp
#         fn = y_total - tp
#         tn = y_reshaped.shape[1] - tp - fp - fn
#
#         self.confusion_matrix += torch.stack([tn, fp, fn, tp], dim=1).reshape(-1, 2, 2).to(self._device)
#
#     def compute(self):
#         if self.normalized:
#             conf = self.confusion_matrix.to(dtype=torch.float64)
#             sums = conf.sum(dim=(1, 2))
#             return conf / sums[:, None, None]
#
#         return self.confusion_matrix
#
#     def _check_input(self, output):
#         y_pred, y = output[0].detach(), output[1].detach()
#
#         if y_pred.ndimension() < 2:
#             raise ValueError(
#                 f"y_pred must at least have shape (batch_size, num_classes (currently set to {self.num_classes}), ...)"
#             )
#
#         if y.ndimension() < 2:
#             raise ValueError(
#                 f"y must at least have shape (batch_size, num_classes (currently set to {self.num_classes}), ...)"
#             )
#
#         if y_pred.shape[0] != y.shape[0]:
#             raise ValueError(f"y_pred and y have different batch size: {y_pred.shape[0]} vs {y.shape[0]}")
#
#         if y_pred.shape[1] != self.num_classes:
#             raise ValueError(f"y_pred does not have correct number of classes: {y_pred.shape[1]} vs {self.num_classes}")
#
#         if y.shape[1] != self.num_classes:
#             raise ValueError(f"y does not have correct number of classes: {y.shape[1]} vs {self.num_classes}")
#
#         if y.shape != y_pred.shape:
#             raise ValueError("y and y_pred shapes must match.")
#
#         valid_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
#         if y_pred.dtype not in valid_types:
#             raise ValueError(f"y_pred must be of any type: {valid_types}")
#
#         if y.dtype not in valid_types:
#             raise ValueError(f"y must be of any type: {valid_types}")
#
#         if not torch.equal(y_pred, y_pred**2):
#             raise ValueError("y_pred must be a binary tensor")
#
#         if not torch.equal(y, y**2):
#             raise ValueError("y must be a binary tensor")