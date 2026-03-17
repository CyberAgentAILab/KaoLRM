import torch
import torch.nn as nn

__all__ = ["LandmarkLoss"]


class LandmarkLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.HuberLoss()
        self.eye_indices = [i for i in range(17, 48)]
        self.mouth_indices = [i for i in range(48, 68)]

    def forward(self, lmks, lmks_gt):
        """
        lmks & lmks_gt: (B, V, 68, 2)
        """
        mask = lmks_gt > 0.0
        valid_mask_BV = (mask).all(dim=(2, 3)).nonzero(as_tuple=True)

        valid_lmks = lmks[valid_mask_BV[0], valid_mask_BV[1]]
        valid_gt_lmks = lmks_gt[valid_mask_BV[0], valid_mask_BV[1]]
        loss_all = self.huber(valid_lmks, valid_gt_lmks)

        valid_eye_lmks = valid_lmks[:, self.eye_indices]
        valid_gt_eye_lmks = valid_gt_lmks[:, self.eye_indices]
        loss_eye = self.huber(valid_eye_lmks, valid_gt_eye_lmks)

        valid_mouth_lmks = valid_lmks[:, self.mouth_indices]
        valid_gt_mouth_lmks = valid_gt_lmks[:, self.mouth_indices]
        loss_mouth = self.huber(valid_mouth_lmks, valid_gt_mouth_lmks)

        return loss_all + loss_mouth + loss_eye


if __name__ == "__main__":
    lmks_pred = torch.rand(16, 4, 5000, 2)
    lmks_gt = torch.rand(16, 4, 5000, 2)
    lmks_gt[3, 0, ...] = -1
    lmks_gt[2, 2, ...] = -1

    criteria = LandmarkLoss()

    loss = criteria(lmks_pred, lmks_gt)
