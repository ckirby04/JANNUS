"""
SC1 loss: penalizes missed lesion objects weighted by volume.

A lesion object is "detected" if the predicted soft mask overlaps the
ground truth connected component with soft IoU >= detection_threshold.
Larger missed lesions incur exponentially higher penalty.
"""

import cc3d
import numpy as np
import torch
import torch.nn as nn


class LesionSensitivityLoss(nn.Module):

    def __init__(self, detection_threshold: float = 0.10,
                 volume_penalty_exponent: float = 1.5):
        super().__init__()
        self.detection_threshold = detection_threshold
        self.volume_penalty_exponent = volume_penalty_exponent

    def forward(self, pred_proba: torch.Tensor,
                gt_mask: torch.Tensor) -> torch.Tensor:
        device = pred_proba.device

        gt_np = gt_mask.detach().cpu().numpy().astype(np.float32)
        if gt_np.ndim > 3:
            gt_np = gt_np.squeeze()
        gt_binary = (gt_np > 0.5).astype(np.uint8)

        labels = cc3d.connected_components(gt_binary, connectivity=26)
        n_lesions = labels.max()

        if n_lesions == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        pred = pred_proba
        if pred.ndim > 3:
            pred = pred.squeeze()
        gt_t = gt_mask
        if gt_t.ndim > 3:
            gt_t = gt_t.squeeze()

        total_penalty = torch.tensor(0.0, device=device)
        labels_t = torch.from_numpy(labels.astype(np.int32)).to(device)

        for i in range(1, n_lesions + 1):
            lesion_mask = (labels_t == i).float()
            lesion_volume = lesion_mask.sum().item()

            pred_in_region = pred * lesion_mask
            gt_in_region = gt_t * lesion_mask

            intersection = (pred_in_region * gt_in_region).sum()
            union = pred_in_region.sum() + gt_in_region.sum() - intersection
            soft_iou = intersection / (union + 1e-6)

            miss_score = torch.clamp(1.0 - soft_iou / self.detection_threshold, min=0.0)

            volume_weight = lesion_volume ** self.volume_penalty_exponent
            total_penalty = total_penalty + miss_score * volume_weight

        max_weight = sum(
            (np.sum(labels == i)) ** self.volume_penalty_exponent
            for i in range(1, n_lesions + 1)
        )
        return total_penalty / (max_weight + 1e-6)
