import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as skmetrics

def calculate_area(pred, label, num_classes, ignore_index=255):
    """
    Calculate intersect, prediction and label area

    Args:
        pred (Tensor): The prediction by model.
        label (Tensor): The ground truth of image.
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.

    Returns:
        Tensor: The intersection area of prediction and the ground on all class.
        Tensor: The prediction area on all class.
        Tensor: The ground truth area on all class
    """
    if len(pred.shape) == 4:
        pred = pred.squeeze(1)
    if len(label.shape) == 4:
        label = label.squeeze(1)
    if not pred.shape == label.shape:
        raise ValueError('Shape of `pred` and `label` should be equal, '
                         'but they are {} and {}.'.format(pred.shape, label.shape))

    pred_area = torch.zeros(num_classes, dtype=torch.long, device=pred.device)
    label_area = torch.zeros(num_classes, dtype=torch.long, device=label.device)
    intersect_area = torch.zeros(num_classes, dtype=torch.long, device=pred.device)
    mask = label != ignore_index

    for i in range(num_classes):
        pred_i = (pred == i) & mask
        label_i = label == i
        intersect_i = pred_i & label_i
        pred_area[i] = pred_i.int().sum(dtype=torch.long)
        label_area[i] = label_i.int().sum(dtype=torch.long)
        intersect_area[i] = intersect_i.int().sum(dtype=torch.long)

    return intersect_area, pred_area, label_area

def auc_roc(logits, label, num_classes, ignore_index=None):
    """
    Calculate area under the roc curve

    Args:
        logits (Tensor): The prediction by model on testset, of shape (N,C,H,W) .
        label (Tensor): The ground truth of image.   (N,C,H,W)
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.

    Returns:
        auc_roc(float): The area under roc curve
    """
    if ignore_index or len(torch.unique(label)) > num_classes:
        raise RuntimeError('labels with ignore_index is not supported yet.')

    if len(label.shape) != 4:
        raise ValueError(
            'The shape of label is not 4 dimension as (N, C, H, W), it is {}'.
                format(label.shape))

    if len(logits.shape) != 4:
        raise ValueError(
            'The shape of logits is not 4 dimension as (N, C, H, W), it is {}'.
                format(logits.shape))

    N, C, H, W = logits.shape
    logits = logits.permute(1, 0, 2, 3).reshape(C, -1).permute(1, 0).float()
    label = label.argmax(dim=1).reshape(N * H * W).cpu().numpy()
    logits = torch.softmax(logits, dim=1).cpu().numpy()

    if not logits.shape[0] == label.shape[0]:
        raise ValueError('length of `logit` and `label` should be equal, '
                         'but they are {} and {}.'.format(logits.shape[0],
                                                          label.shape[0]))

    if num_classes == 2:
        auc = skmetrics.roc_auc_score(label, logits[:, 1])
    else:
        auc = skmetrics.roc_auc_score(label, logits, multi_class='ovr')

    return auc


def mean_iou(intersect_area, pred_area, label_area):
    """
    Calculate iou.

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.

    Returns:
        np.ndarray: iou on all classes.
        float: mean iou of all classes.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    union = pred_area + label_area - intersect_area
    class_iou = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            iou = 0
        else:
            iou = intersect_area[i] / union[i]
        class_iou.append(iou)
    miou = np.mean(class_iou)
    return np.array(class_iou), miou

def dice(intersect_area, pred_area, label_area):
    """
    Calculate DICE.

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.

    Returns:
        np.ndarray: DICE on all classes.
        float: mean DICE of all classes.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    union = pred_area + label_area
    class_dice = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            dice = 0
        else:
            dice = (2 * intersect_area[i]) / union[i]
        class_dice.append(dice)
    mdice = np.mean(class_dice)
    return np.array(class_dice), mdice


def accuracy(intersect_area, pred_area):
    """
    Calculate accuracy

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes..
        pred_area (Tensor): The prediction area on all classes.

    Returns:
        np.ndarray: accuracy on all classes.
        float: mean accuracy.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    class_acc = []
    for i in range(len(intersect_area)):
        if pred_area[i] == 0:
            acc = 0
        else:
            acc = intersect_area[i] / pred_area[i]
        class_acc.append(acc)
    macc = np.sum(intersect_area) / np.sum(pred_area)
    return np.array(class_acc), macc


def kappa(intersect_area, pred_area, label_area):
    """
    Calculate kappa coefficient

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes..
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.

    Returns:
        float: kappa coefficient.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    total_area = np.sum(label_area)
    po = np.sum(intersect_area) / total_area
    pe = np.sum(pred_area * label_area) / (total_area * total_area)
    kappa = (po - pe) / (1 - pe)
    return kappa

class InferTestset(nn.Module):
    def __init__(self, num_classes, reduction='none', ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, pred, ture):
        intersect_area, pred_area, label_area = calculate_area(pred, ture, self.num_classes, self.ignore_index)
        class_iou, miou = mean_iou(intersect_area, pred_area, label_area)
        class_dice, mdice = dice(intersect_area, pred_area, label_area)
        class_acc, macc = accuracy(intersect_area, pred_area)
        kappa_coeff = kappa(intersect_area, pred_area, label_area)
        auc = auc_roc(logits, label, self.num_classes)
        if self.reduction == 'mean':
            return miou, mdice, macc, kappa_coeff, auc
        return class_iou, class_dice, class_acc, kappa_coeff, auc

    def get_unique(self, pred, ture):
        return torch.unique(pred), torch.unique(ture)

if __name__=="__main__":
    num_classes = 3
    batch_size = 1
    height, width = 4, 4
    # logits = torch.randint(0, num_classes, (batch_size, height, width))
    # logits = torch.nn.functional.one_hot(logits, num_classes=num_classes).permute(0, 3, 1, 2)
    # label = torch.randint(0, num_classes, (batch_size, height, width))
    # label = torch.nn.functional.one_hot(label, num_classes=num_classes).permute(0, 3, 1, 2)
    #
    # print(logits.shape, label.shape)
    logits = torch.Tensor([[[[1, 2, 2, 0],
                             [1, 0, 2, 2],
                             [1, 0, 0, 1],
                             [0, 1, 0, 2]],

                            [[1, 1, 1, 2],
                             [1, 2, 0, 2],
                             [0, 2, 1, 1],
                             [1, 0, 0, 0]],

                            [[1, 0, 2, 0],
                             [0, 2, 0, 2],
                             [0, 2, 2, 0],
                             [2, 0, 0, 0]]]])
    #
    label = torch.Tensor([[[[1, 0, 2, 0],
                             [2, 0, 2, 2],
                             [1, 1, 0, 1],
                             [0, 0, 0, 2]],

                            [[2, 1, 1, 2],
                             [1, 1, 0, 2],
                             [0, 2, 1, 1],
                             [1, 0, 0, 1]],

                            [[1, 0, 2, 2],
                             [1, 2, 2, 2],
                             [0, 2, 2, 0],
                             [1, 0, 0, 0]]]])

    # logits = logits.argmax(dim=1)
    eval_test = InferTestset(num_classes, reduction='none')
    class_iou, class_dice, class_acc, kappa_coeff, auc = eval_test(logits, label)

    eval_test = InferTestset(num_classes, reduction='mean')
    miou, mdice, macc, _kappa_coeff, _auc = eval_test(logits, label)
    print(eval_test.get_unique(logits, label))

    # Print metrics
    print("Class IOU:", class_iou)
    print("Mean IOU:", miou)
    print("Class DICE:", class_dice)
    print("Mean DICE:", mdice)
    print("Class Accuracy:", class_acc)
    print("Mean Accuracy:", macc)
    print("Kappa Coefficient:", kappa_coeff)
    print("AUC-ROC:", auc)
