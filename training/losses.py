import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss for imbalanced fatigue & yawning classification.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MultimodalDrowsinessLoss(nn.Module):
    """
    Joint loss combining Classification Cross-Entropy, Focal Loss, and Sequence Smoothness.
    """
    def __init__(self, use_focal: bool = True):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.focal = FocalLoss() if use_focal else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.ce(logits, targets)
        if self.focal is not None:
            loss = loss + 0.5 * self.focal(logits, targets)
        return loss
