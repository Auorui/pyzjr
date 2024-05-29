"""
compute_sdf1_1、compute_sdf、boundary_loss
Reference from https://github.com/LIVIAETS/boundary-loss?tab=readme-ov-file
time 2024-01-26
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage import segmentation as skimage_seg

def compute_sdf1_1(img_gt, out_shape):
    """
    compute the normalized signed distance map of binary mask
    input: segmentation, shape = (batch_size, c, x, y)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1, 1]
    """
    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        # ignore background
        posmask = img_gt[b].astype(bool)
        for c in range(out_shape[1]):
            if posmask.any():
                negmask = ~posmask
                posdis = distance_transform_edt(posmask)
                negdis = distance_transform_edt(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (
                            np.max(posdis) - np.min(posdis))
                sdf[boundary == 1] = 0
                normalized_sdf[b][c] = sdf

    return normalized_sdf


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """
    img_gt = img_gt.astype(np.uint8)
    gt_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        posmask = img_gt[b].astype(bool)
        for c in range(out_shape[1]):
            if posmask.any():
                negmask = ~posmask
                posdis = distance_transform_edt(posmask)
                negdis = distance_transform_edt(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = negdis - posdis
                sdf[boundary == 1] = 0
                gt_sdf[b][c] = sdf

    return gt_sdf

def boundary_loss(outputs_soft, label_batch):
    """
    compute boundary loss for binary segmentation
    Args:
        outputs_soft: sigmoid results,  shape=(b,c,x,y)
        label_batch: sdf of ground truth (can be original or normalized sdf); shape=(b,c,x,y)

    Returns: boundary_loss
    """
    gt_sdf_npy = compute_sdf1_1(label_batch.numpy(), outputs_soft.shape)
    gt_sdf = torch.from_numpy(gt_sdf_npy).float()
    pc = outputs_soft.type(torch.float32)
    dc = gt_sdf.type(torch.float32)

    multipled = torch.einsum("bcwh, bcwh->bcwh", pc, dc)
    loss = multipled.mean(dim=(1, 2, 3))

    return loss

def softmax_focal_loss_3d(input, target, gamma=2.0, alpha=None):
    """
    FL(pt) = -alpha * (1 - pt)**gamma * log(pt)

    where p_i = exp(s_i) / sum_j exp(s_j), t is the target (ground truth) class, and
    s_j is the unnormalized score for class j.
    """
    input_ls = input.log_softmax(1)
    loss = -(1 - input_ls.exp()).pow(gamma) * input_ls * target
    if alpha is not None:
        # (1-alpha) for the background class and alpha for the other classes
        alpha_fac = torch.tensor([1 - alpha] + [alpha] * (target.shape[1] - 1)).to(loss)
        broadcast_dims = [-1] + [1] * len(target.shape[2:])
        alpha_fac = alpha_fac.view(broadcast_dims)
        loss = alpha_fac * loss

    return loss

def sigmoid_focal_loss_3d(input, target, gamma=2.0, alpha=None):
    """
    FL(pt) = -alpha * (1 - pt)**gamma * log(pt)
    where p = sigmoid(x), pt = p if label is 1 or 1 - p if label is 0
    """
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    invprobs = F.logsigmoid(-input * (target * 2 - 1))  # reduced chance of overflow
    loss = (invprobs * gamma).exp() * loss

    if alpha is not None:
        # alpha if t==1; (1-alpha) if t==0
        alpha_factor = target * alpha + (1 - target) * (1 - alpha)
        loss = alpha_factor * loss

    return loss


if __name__=="__main__":
    batch_size, channels, height, width = 1, 1, 3, 3
    # model_predictions = torch.rand((batch_size, channels, height, width))
    model_predictions = torch.Tensor([[[[0.7616, 0.0239, 0.5281],
                                        [0.3337, 0.5539, 0.3333],
                                        [0.1318, 0.8004, 0.5439]]]])

    model_predictions = torch.sigmoid(model_predictions)
    seg_mask = np.random.randint(2, size=(batch_size, channels, height, width))
    seg_mask_tensor = torch.from_numpy(seg_mask).float()
    loss = boundary_loss(model_predictions, seg_mask_tensor)

    # Print the loss value
    print("Boundary Loss:", loss.item())
