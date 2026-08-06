"""SC2 loss: standard soft Dice."""

import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_proba: torch.Tensor,
                gt_mask: torch.Tensor) -> torch.Tensor:
        pred_flat = pred_proba.reshape(-1)
        gt_flat = gt_mask.reshape(-1)

        intersection = (pred_flat * gt_flat).sum()
        denominator = pred_flat.sum() + gt_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice
