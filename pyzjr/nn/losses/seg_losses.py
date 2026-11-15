"""
Copyright (c) 2024, Auorui.
All rights reserved.
time 2024-01-25
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries

class _SegLossBase(nn.Module):
    def __init__(
            self,
            num_classes=2,
            ignore_index=256,
            reduction="mean",
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")

        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"Unsupported reduction: {reduction}")

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reduction = reduction

    def _prepare_input_target(self, input, target):
        if input.dim() != 4:
            raise ValueError(
                f"input must be (B,C,H,W), got {input.shape}"
            )

        if input.size(1) != self.num_classes:
            raise ValueError(
                f"input channels={input.size(1)} does not match "
                f"num_classes={self.num_classes}"
            )

        if target.dim() == 4:
            if target.size(1) != 1:
                raise ValueError(
                    f"target must be (B,1,H,W), got {target.shape}"
                )
            target = target[:, 0]

        if target.dim() != 3:
            raise ValueError(
                f"target must be (B,H,W) or (B,1,H,W), "
                f"got {target.shape}"
            )

        target = target.long()

        if input.shape[-2:] != target.shape[-2:]:
            input = F.interpolate(
                input,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if self.ignore_index is None:
            valid_mask = torch.ones_like(
                target,
                dtype=torch.bool,
            )
        else:
            valid_mask = target != self.ignore_index

        invalid = valid_mask & (
                (target < 0) |
                (target >= self.num_classes)
        )

        if invalid.any():
            invalid_values = torch.unique(target[invalid]).tolist()
            raise ValueError(
                f"target contains invalid class ids: {invalid_values}"
            )

        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        return input, target, target_safe, valid_mask

    def _reduce_pixel_loss(self, pixel_loss, valid_mask):
        """
        pixel_loss: (B,H,W)
        valid_mask: (B,H,W)

        reduction='none' 返回每个样本一个损失，形状为 (B)。
        """
        valid_mask = valid_mask.to(dtype=pixel_loss.dtype)

        loss_sum = (
                pixel_loss * valid_mask
        ).flatten(1).sum(dim=1)

        valid_count = valid_mask.flatten(1).sum(dim=1)

        loss_per_sample = loss_sum / valid_count.clamp_min(1.0)

        # 整个样本都是 ignore_index 时，损失置 0
        loss_per_sample = torch.where(
            valid_count > 0,
            loss_per_sample,
            torch.zeros_like(loss_per_sample),
            )

        if self.reduction == "none":
            return loss_per_sample

        if self.reduction == "mean":
            return loss_per_sample.mean()

        return loss_per_sample.sum()

class _OverlapLossBase(_SegLossBase):
    def __init__(
            self,
            num_classes=2,
            smooth=1e-5,
            ignore_index=256,
            reduction="mean",
            include_background=False,
    ):
        super().__init__(
            num_classes=num_classes,
            ignore_index=ignore_index,
            reduction=reduction,
        )

        self.smooth = smooth
        self.include_background = include_background

    def _prepare_overlap(self, input, target):
        input, _, target_safe, valid_mask = \
            self._prepare_input_target(input, target)

        prob = F.softmax(input, dim=1)

        target_one_hot = F.one_hot(
            target_safe,
            num_classes=self.num_classes,
        ).permute(0, 3, 1, 2).to(dtype=input.dtype)

        mask = valid_mask.unsqueeze(1).to(dtype=input.dtype)

        prob = prob * mask
        target_one_hot = target_one_hot * mask

        if self.num_classes == 2:
            if self.include_background:
                class_ids = [0, 1]
            else:
                class_ids = [1]
        else:
            if self.include_background:
                class_ids = list(range(self.num_classes))
            else:
                class_ids = list(range(1, self.num_classes))

        prob = prob[:, class_ids]
        target_one_hot = target_one_hot[:, class_ids]

        intersection = (
                prob * target_one_hot
        ).sum(dim=(2, 3))

        prob_sum = prob.sum(dim=(2, 3))
        target_sum = target_one_hot.sum(dim=(2, 3))

        valid_pixels = valid_mask.sum(dim=(1, 2))

        class_valid = ((valid_pixels > 0).unsqueeze(1) & ((prob_sum + target_sum) > 0))

        return (
            intersection,
            prob_sum,
            target_sum,
            class_valid,
        )

    def _reduce_class_loss(self, class_loss, class_valid):
        class_loss = (
                class_loss
                * class_valid.to(dtype=class_loss.dtype)
        )

        valid_class_count = \
            class_valid.sum(dim=1).clamp_min(1)

        loss = (class_loss.sum(dim=1) / valid_class_count)

        loss = torch.where(
            class_valid.any(dim=1),
            loss,
            torch.zeros_like(loss),
        )

        if self.reduction == "none":
            return loss

        if self.reduction == "mean":
            return loss.mean()

        return loss.sum()

class CrossEntropyLoss(_SegLossBase):
    def __init__(
            self,
            num_classes=2,
            weight=None,
            reduction="mean",
            ignore_index=256,
    ):
        super().__init__(num_classes=num_classes, ignore_index=ignore_index, reduction=reduction)

        if weight is not None:
            weight = torch.as_tensor(weight, dtype=torch.float32,)
            if weight.numel() != num_classes:
                raise ValueError(
                    "weight length must equal num_classes"
                )
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, input, target):
        input, target, _, valid_mask = \
            self._prepare_input_target(input, target)

        if self.ignore_index is None:
            pixel_loss = F.cross_entropy(
                input,
                target,
                weight=self.weight,
                reduction="none",
            )
        else:
            pixel_loss = F.cross_entropy(
                input,
                target,
                weight=self.weight,
                ignore_index=self.ignore_index,
                reduction="none",
            )

        return self._reduce_pixel_loss(pixel_loss, valid_mask)

class FocalLoss(_SegLossBase):
    def __init__(
            self,
            num_classes=2,
            alpha=0.25,
            gamma=2.0,
            reduction="mean",
            ignore_index=256,
    ):
        super().__init__(num_classes=num_classes, ignore_index=ignore_index, reduction=reduction,)
        if gamma < 0:
            raise ValueError("gamma must be >= 0")

        self.alpha = alpha
        self.gamma = gamma

    def _get_alpha_factor(self, target_safe, valid_mask, dtype):
        if self.alpha is None:
            return torch.ones_like(target_safe, dtype=dtype)

        if isinstance(self.alpha, (int, float)):
            alpha = float(self.alpha)

            if not 0.0 <= alpha <= 1.0:
                raise ValueError(
                    "scalar alpha must be in [0, 1]"
                )

            if self.num_classes == 2:
                # 二分类：前景 alpha，背景 1-alpha
                alpha_factor = torch.where(
                    target_safe == 1,
                    torch.full_like(
                        target_safe,
                        alpha,
                        dtype=dtype,
                    ),
                    torch.full_like(
                        target_safe,
                        1.0 - alpha,
                        dtype=dtype,
                        ),
                    )
            else:
                # 多分类：背景 1-alpha，所有前景 alpha
                alpha_factor = torch.where(
                    target_safe == 0,
                    torch.full_like(
                        target_safe,
                        1.0 - alpha,
                        dtype=dtype,
                        ),
                    torch.full_like(
                        target_safe,
                        alpha,
                        dtype=dtype,
                    ),
                    )

        else:
            alpha_vector = torch.as_tensor(self.alpha, dtype=dtype, device=target_safe.device)

            if alpha_vector.numel() != self.num_classes:
                raise ValueError(
                    "alpha length must equal num_classes"
                )

            alpha_factor = alpha_vector[target_safe]

        return alpha_factor * valid_mask.to(dtype=dtype)

    def forward(self, input, target):
        input, target, target_safe, valid_mask = \
            self._prepare_input_target(input, target)

        if self.ignore_index is None:
            ce_loss = F.cross_entropy(
                input,
                target,
                reduction="none",
            )
        else:
            ce_loss = F.cross_entropy(
                input,
                target,
                ignore_index=self.ignore_index,
                reduction="none",
            )

        pt = torch.exp(-ce_loss)

        alpha_factor = self._get_alpha_factor(
            target_safe=target_safe,
            valid_mask=valid_mask,
            dtype=ce_loss.dtype,
        )

        focal_loss = (
                alpha_factor
                * (1.0 - pt).pow(self.gamma)
                * ce_loss
        )

        return self._reduce_pixel_loss(
            focal_loss,
            valid_mask,
        )


class DiceLoss(_OverlapLossBase):
    def forward(self, input, target):
        intersection, prob_sum, target_sum, class_valid = \
            self._prepare_overlap(input, target)

        denominator = prob_sum + target_sum

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return self._reduce_class_loss(1.0 - dice, class_valid)


class JaccardLoss(_OverlapLossBase):
    def forward(self, input, target):
        intersection, prob_sum, target_sum, class_valid = \
            self._prepare_overlap(input, target)

        # Jaccard 的并集要减掉 intersection
        union = prob_sum + target_sum - intersection

        jaccard = (intersection + self.smooth) / (union + self.smooth)

        return self._reduce_class_loss(
            1.0 - jaccard,
            class_valid,
            )

class TanimotoLoss(_OverlapLossBase):
    def __init__(
            self,
            num_classes=2,
            smooth=1e-5,
            ignore_index=256,
            reduction="mean",
            include_background=False,
            use_complement=True,
    ):
        super().__init__(
            num_classes=num_classes,
            smooth=smooth,
            ignore_index=ignore_index,
            reduction=reduction,
            include_background=include_background,
        )

        self.use_complement = use_complement

    def forward(self, input, target):
        input, _, target_safe, valid_mask = \
            self._prepare_input_target(input, target)

        prob = F.softmax(input, dim=1)

        target_one_hot = F.one_hot(
            target_safe,
            num_classes=self.num_classes,
        ).permute(0, 3, 1, 2).to(dtype=input.dtype)

        mask = valid_mask.unsqueeze(1).to(dtype=input.dtype)

        if self.num_classes == 2:
            if self.include_background:
                class_ids = [0, 1]
            else:
                # 二分类默认只计算前景
                class_ids = [1]
        else:
            if self.include_background:
                class_ids = list(range(self.num_classes))
            else:
                # 多分类默认忽略背景
                class_ids = list(range(1, self.num_classes))

        prob = prob[:, class_ids]
        target_one_hot = target_one_hot[:, class_ids]

        # 注意：这里不要提前 prob = prob * mask
        # 每个统计项单独乘 mask，避免 ignore 区域进入 complement 分支
        intersection = (prob * target_one_hot * mask).sum(dim=(2, 3))

        p2 = (prob * prob * mask).sum(dim=(2, 3))

        t2 = (target_one_hot * target_one_hot * mask).sum(dim=(2, 3))

        tanimoto = (intersection + self.smooth) / (p2 + t2 - intersection + self.smooth)

        if self.use_complement:
            prob_complement = 1.0 - prob
            target_complement = 1.0 - target_one_hot

            intersection_c = (prob_complement * target_complement * mask).sum(dim=(2, 3))

            p2_c = (prob_complement * prob_complement * mask).sum(dim=(2, 3))

            t2_c = (target_complement * target_complement * mask).sum(dim=(2, 3))

            tanimoto_c = (intersection_c + self.smooth) / (p2_c + t2_c - intersection_c + self.smooth)

            tanimoto_score = 0.5 * (tanimoto + tanimoto_c)
        else:
            tanimoto_score = tanimoto

        loss = 1.0 - tanimoto_score

        # 类别有效性判断
        valid_pixels = valid_mask.sum(dim=(1, 2))

        prob_sum = (prob * mask).sum(dim=(2, 3))

        target_sum = (target_one_hot * mask).sum(dim=(2, 3))

        class_valid = ((valid_pixels > 0).unsqueeze(1) & ((prob_sum + target_sum) > 0))

        return self._reduce_class_loss(loss, class_valid)


class BoundaryLoss(nn.Module):
    """
    Boundary Loss (MICCAI 2019).
    Encourages the model to align predictions with the ground truth boundaries
    using a pre-computed Signed Distance Map (SDM).

    Note: This loss computes the SDM on the CPU, which is relatively slow.
    It is recommended to use it as an auxiliary loss (e.g., combined with Dice).
    """
    def __init__(self, num_classes=2, selected_class=1):
        super(BoundaryLoss, self).__init__()
        self.num_classes = num_classes
        self.selected_class = selected_class

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (B, C, H, W) logits
            y_true: (B, H, W) label index map
        """
        #  (B, H, W) --> One-Hot
        if y_true.dim() == 3:
            y_true_one_hot = torch.nn.functional.one_hot(y_true.long(), num_classes=self.num_classes)
            y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2).float()
        else:
            y_true_one_hot = y_true.float()

        # Extract a binary image based on the selected category
        y_true_selected = y_true_one_hot[:, self.selected_class, :, :].unsqueeze(1) # (B, 1, H, W)
        y_pred_selected = torch.sigmoid(y_pred[:, self.selected_class, :, :].unsqueeze(1)) # (B, 1, H, W)

        gt_sdf_npy = self.compute_sdf(y_true_selected.cpu().numpy(), y_pred_selected.shape)
        gt_sdf = torch.from_numpy(gt_sdf_npy).float().to(y_pred.device)

        multipled = torch.einsum("bchw, bchw->bchw", y_pred_selected, gt_sdf)

        return multipled.mean()

    def compute_sdf(self, img_gt, out_shape):
        """计算归一化的有符号距离图 (SDM)"""
        img_gt = img_gt.astype(np.uint8)
        normalized_sdf = np.zeros(out_shape)

        for b in range(out_shape[0]):
            # The shape of img_gt[b] is now (1, H, W)
            posmask = img_gt[b].squeeze(0).astype(bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance_transform_edt(posmask)
                negdis = distance_transform_edt(negmask)
                boundary = find_boundaries(posmask, mode='inner').astype(np.uint8)

                # Normalize to [-1, 1]
                sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - \
                      (posdis - np.min(posdis)) / (np.max(posdis) - np.min(posdis))
                sdf[boundary == 1] = 0
                normalized_sdf[b][0] = sdf

        return normalized_sdf



