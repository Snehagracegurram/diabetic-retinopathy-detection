from __future__ import annotations

import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(logits.size(0), logits.size(1), -1)
        targets = targets.contiguous().view(targets.size(0), targets.size(1), -1)
        inter = (probs * targets).sum(dim=2)
        denom = probs.sum(dim=2) + targets.sum(dim=2)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss
