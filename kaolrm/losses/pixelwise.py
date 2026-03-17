# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn

__all__ = ["PixelLoss"]


class PixelLoss(nn.Module):
    """
    Pixel-wise loss between two images.
    """

    def __init__(self, option: str = "mse"):
        super().__init__()
        self.loss_fn = self._build_from_option(option)

    @staticmethod
    def _build_from_option(option: str, reduction: str = "none"):
        if option == "mse":
            return nn.MSELoss(reduction=reduction)
        elif option == "l1":
            return nn.L1Loss(reduction=reduction)
        else:
            raise NotImplementedError(f"Unknown pixel loss option: {option}")

    @torch.compile
    def forward(self, x, y, mask=None, mask_weight=0.7):
        """
        Assume images are channel first.

        Args:
            x: [N, M, C, H, W]
            y: [N, M, C, H, W]
            mask: [N, M, 1, H, W] or [N, M, H, W] or None

        Returns:
            Mean-reduced pixel loss across batch.
        """
        N, M, C, H, W = x.shape
        x = x.reshape(N * M, C, H, W)
        y = y.reshape(N * M, C, H, W)

        pixel_loss = self.loss_fn(x, y)  # [N*M, C, H, W]
        image_loss = pixel_loss.mean(dim=[1, 2, 3])  # [N*M]

        if mask is not None:
            # Handle different possible mask shapes
            if mask.dim() == 4:  # [N, M, H, W]
                mask = mask.unsqueeze(2)  # [N, M, 1, H, W]
            elif mask.dim() == 5 and mask.shape[2] == 1:
                pass  # already correct
            else:
                raise ValueError(f"Mask must have shape [N, M, H, W] or [N, M, 1, H, W], got {mask.shape}")

            mask = mask.float()  # need float for multiplication
            mask = mask.expand(-1, -1, C, -1, -1)  # [N, M, C, H, W]
            mask = mask.reshape(N * M, C, H, W)  # [N*M, C, H, W]

            # Multiply loss by mask, sum over non-batch dimensions, divide by mask sum (avoid zero division)
            masked_loss = pixel_loss * mask
            masked_sum = masked_loss.sum(dim=(1, 2, 3))  # [N*M]
            mask_sum = mask.sum(dim=(1, 2, 3)).clamp(min=1e-6)  # [N*M], avoid division by zero
            image_loss = (1.0 - mask_weight) * image_loss + mask_weight * masked_sum / mask_sum  # [N*M]

        batch_loss = image_loss.reshape(N, M).mean(dim=1)  # [N]
        all_loss = batch_loss.mean()
        return all_loss
