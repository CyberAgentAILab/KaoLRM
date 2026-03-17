import torch
import torch.nn as nn

__all__ = ["GeometryLoss"]


class GeometryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask=None):
        """
        pred, gt:      [B, V, C, H, W]
        mask:          [B, V, 1, H, W] or [B, V, H, W]
        """
        # pred, gt = pred[:,:1,...], gt[:,:1,...]
        loss = torch.abs(pred - gt)  # [B, V, C, H, W]
        # mask broadcast: [B, V, 1, H, W]  or unsqueeze(2) if mask is [B, V, H, W]
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.unsqueeze(2)
            masked_loss = loss * mask  # [B, V, C, H, W]
            valid_count = mask.sum() * pred.size(2)  # if you want to average per-pixel, not per-channel, sum(dim=2) first
            return masked_loss.sum() / (valid_count + 1e-8)

        else:
            return loss.mean()


if __name__ == "__main__":
    pred = torch.rand([16, 3, 1, 288, 288])
    gt = torch.rand([16, 3, 1, 288, 288])
    mask = torch.rand([16, 3, 1, 288, 288])
    criteria = GeometryLoss()

    loss = criteria(pred, gt, mask)
    print(loss)
