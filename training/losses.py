import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss for imbalanced fatigue & drowsiness classification.
    Down-weights easy examples so the model focuses on hard mis-classified ones.
    gamma=2 is standard; alpha can be per-class weight tensor.
    """
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        # alpha can be a scalar or per-class weight Tensor
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-Entropy with label smoothing to prevent overconfident predictions
    and improve calibration. smoothing=0.1 is standard.
    """
    def __init__(self, num_classes: int = 5, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)
        # Hard target loss
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        # Smooth target loss
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class MultimodalDrowsinessLoss(nn.Module):
    """
    Joint loss: Label-Smoothed CE + Focal Loss + optional binary fatigue BCE.
    Combining these handles class imbalance and improves generalisation on NTHU-DDD.
    """
    def __init__(
        self,
        num_classes: int = 5,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        focal_weight: float = 0.5,
        class_weights: torch.Tensor = None
    ):
        super().__init__()
        self.label_smooth_ce = LabelSmoothingCrossEntropy(num_classes=num_classes, smoothing=label_smoothing)
        self.focal = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.focal_weight = focal_weight
        # Binary fatigue BCE (class 4 = drowsy/eye_closure is "drowsy" positive)
        self.bce = nn.BCELoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        fatigue_score: torch.Tensor = None
    ) -> torch.Tensor:
        # 1. Label-smoothed Cross Entropy (primary)
        loss = self.label_smooth_ce(logits, targets)
        # 2. Focal Loss (handles class imbalance)
        loss = loss + self.focal_weight * self.focal(logits, targets)
        # 3. Binary fatigue consistency (optional auxiliary head)
        if fatigue_score is not None:
            binary_label = (targets >= 1).float().unsqueeze(1)  # 0=normal, 1=any drowsy
            with torch.amp.autocast('cuda', enabled=False):
                loss = loss + 0.2 * F.binary_cross_entropy(
                    fatigue_score.float().clamp(1e-7, 1.0 - 1e-7),
                    binary_label.float()
                )
        return loss
