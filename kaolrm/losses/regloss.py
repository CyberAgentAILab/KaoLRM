import torch
import torch.nn as nn

__all__ = ["L2RegLoss", "ThresRegLoss"]


class L2RegLoss(nn.Module):
    """
    Regularization on shape & expression params of FLAME
    """

    def __init__(self):
        super().__init__()

    def forward(self, params):
        reg_loss = torch.sum(params**2)  # ||W||^2

        return reg_loss


class ThresRegLoss(nn.Module):
    """
    Regularization on 3D Gaussians' scales over a given threshold.
    """

    def __init__(self, threshold=0.06):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        reg_loss = torch.sum(torch.relu(x - self.threshold) ** 2)
        return reg_loss
