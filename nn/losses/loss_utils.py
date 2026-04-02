import torch
import torch.nn as nn
from typing import Tuple

class JointLoss(nn.Module):
    def __init__(self, *criterions, loss_weight:Tuple[int, float, float]):
        """Weighted summation of combining multiple loss functions

        Args:
            *criterions (nn.Module): Multiple loss function modules
            loss_weight (Tuple[int, float, float]): Corresponding weights for each loss function
        """
        super().__init__()
        if len(criterions) != len(loss_weight):
            raise ValueError(f"Number of criterions ({len(criterions)}) "
                             f"must match weights count ({len(loss_weight)})")
        for i, criterion in enumerate(criterions):
            self.add_module(f'criterion_{i}', criterion)

        self.register_buffer('loss_weights',
                             torch.tensor(loss_weight, dtype=torch.float32))

    def forward(self, y_pred, y_true):
        total_loss = 0.0
        for i, (weight, criterion) in enumerate(
                zip(self.loss_weights, self.children())
        ):
            loss = criterion(y_pred, y_true)
            total_loss += weight * loss

        return total_loss

if __name__=="__main__":
    from pyzjr.nn.losses.loss_function import CrossEntropyLoss, JaccardLoss, FocalLoss
    y_pred = torch.rand(4, 3, 5, 5)
    y_true = torch.randint(0, 2, (4, 3, 5, 5)).float()
    loss11 = JointLoss(CrossEntropyLoss(), JaccardLoss(), FocalLoss(),
                       loss_weight=(1, 0.2, 0.1))(y_pred, y_true)
    print(loss11)
