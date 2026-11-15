import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """
    A flexible loss container that combines multiple loss functions with weights.

    Args:
        loss_dict (dict): A dictionary mapping loss names to their instantiated modules or callables.
                          e.g., {"ce": nn.CrossEntropyLoss(), "dice": DiceLoss()}
        loss_type (list): A list of loss names to use and their order.
        loss_weight (list): A list of weights corresponding to each loss in `loss_type`.
        aux_loss_weights (list, optional): Weights for auxiliary branches (deep supervision).
    """
    def __init__(self, loss_dict, loss_type=None, loss_weight=None, aux_loss_weights=None, reduction='mean'):
        super(CombinedLoss, self).__init__()

        self.loss_map = nn.ModuleDict(loss_dict)
        self.loss_type = loss_type if loss_type is not None else list(loss_dict.keys())
        self.loss_weight = loss_weight if loss_weight is not None else [1.0] * len(self.loss_type)
        self.reduction = reduction
        self.aux_loss_weights = aux_loss_weights

        assert len(self.loss_type) == len(self.loss_weight), \
            "The lengths of loss_type and loss_weight must be consistent."

        for name in self.loss_type:
            if name not in self.loss_map:
                raise ValueError(f"Loss type '{name}' is not registered in the loss_dict.")

    def _compute_single_loss(self, input, target):
        total_loss = 0.0

        for loss_name, weight in zip(self.loss_type, self.loss_weight):
            total_loss = total_loss + weight * self.loss_map[loss_name](input, target)

        return total_loss

    def forward(self, input, target):
        """
        Args:
            input: It can be a single tensor of shape (B, C, H, W) or a list/tuple of multiple tensors
            target: (B, H, W) or (B, 1, H, W) labels
        Returns:
            loss: scalar
        """
        if isinstance(input, (list, tuple)):
            losses = [self._compute_single_loss(inp, target) for inp in input]
            if self.aux_loss_weights and len(self.aux_loss_weights) == len(losses):
                total_loss = sum(w * l for w, l in zip(self.aux_loss_weights, losses))
            else:
                total_loss = sum(losses)
            return total_loss
        else:
            return self._compute_single_loss(input, target)


if __name__ == "__main__":
    from pyzjr.nn.losses.seg_losses import CrossEntropyLoss, JaccardLoss, FocalLoss, DiceLoss, BoundaryLoss, TanimotoLoss

    print("=" * 50)
    print("1. 测试基础二分类组合 (CE + Dice)")
    print("=" * 50)
    y_pred = torch.randn(2, 2, 64, 64)
    y_true = torch.randint(0, 2, (2, 64, 64)).long()

    criterion_1 = CombinedLoss(
        loss_dict={
            "ce": CrossEntropyLoss(num_classes=2),
            "dice": DiceLoss(num_classes=2),
        },
        loss_type=["ce", "dice"],
        loss_weight=[0.5, 1.0]
    )
    loss_1 = criterion_1(y_pred, y_true)
    print(f"Loss 1: {loss_1.item():.4f}")
    assert loss_1.dim() == 0, "Loss should be a scalar"

    print("\n" + "=" * 50)
    print("2. 测试多分支深监督网络 (List Input)")
    print("=" * 50)
    # 模拟深监督：主分支 和 辅助分支 (例如 DeepLabV3+ 的 Aux Head)
    y_pred_main = torch.randn(2, 2, 64, 64)
    y_pred_aux = torch.randn(2, 2, 64, 64)

    criterion_2 = CombinedLoss(
        loss_dict={
            "ce": CrossEntropyLoss(num_classes=2),
            "dice": DiceLoss(num_classes=2),
        },
        loss_type=["ce", "dice"],
        loss_weight=[0.5, 0.5],
        aux_loss_weights=[1.0, 0.4] # 主分支权重1.0，辅助分支权重0.4
    )
    loss_2 = criterion_2([y_pred_main, y_pred_aux], y_true)
    print(f"Loss 2 (with aux): {loss_2.item():.4f}")

    print("\n" + "=" * 50)
    print("3. 测试加入 BoundaryLoss (自定义组装)")
    print("=" * 50)
    criterion_3 = CombinedLoss(
        loss_dict={
            "ce": CrossEntropyLoss(num_classes=2),
            "dice": DiceLoss(num_classes=2),
            "boundary": BoundaryLoss(num_classes=2, selected_class=1),
        },
        loss_type=["ce", "dice", "boundary"],
        loss_weight=[0.5, 0.8, 0.1]
    )
    loss_3 = criterion_3(y_pred, y_true)
    print(f"Loss 3 (with Boundary): {loss_3.item():.4f}")

    print("\n" + "=" * 50)
    print("4. 测试多分类任务 (Focal + Jaccard)")
    print("=" * 50)
    y_pred_mc = torch.randn(2, 4, 64, 64)
    y_true_mc = torch.randint(0, 4, (2, 64, 64)).long()

    criterion_4 = CombinedLoss(
        loss_dict={
            "focal": FocalLoss(num_classes=4, alpha=0.25, gamma=2.0),
            "iou": JaccardLoss(num_classes=4),
        },
        loss_type=["focal", "iou"],
        loss_weight=[1.0, 1.0]
    )
    loss_4 = criterion_4(y_pred_mc, y_true_mc)
    print(f"Loss 4 (multiclass): {loss_4.item():.4f}")