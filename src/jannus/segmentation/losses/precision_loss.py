"""SC3 loss: precision-weighted BCE with explicit false positive penalty."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrecisionLoss(nn.Module):

    def __init__(self, fp_penalty_weight: float = 2.0, smooth: float = 1e-5):
        super().__init__()
        self.fp_penalty_weight = fp_penalty_weight
        self.smooth = smooth

    def forward(self, pred_proba: torch.Tensor,
                gt_mask: torch.Tensor) -> torch.Tensor:
        pred_flat = pred_proba.reshape(-1)
        gt_flat = gt_mask.reshape(-1)

        bce = F.binary_cross_entropy(pred_flat, gt_flat, reduction="mean")
        fp_term = (pred_flat * (1.0 - gt_flat)).mean()

        return bce + self.fp_penalty_weight * fp_term
