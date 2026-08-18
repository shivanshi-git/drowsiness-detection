import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Classification and Imbalanced/Hard Sample Learning.
    
    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        
    Parameters:
        alpha (float or torch.Tensor): Weighting factor for positive/negative class (default 0.25).
        gamma (float): Focusing parameter for hard samples (default 2.0).
        reduction (str): 'mean', 'sum', or 'none'.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if isinstance(self.alpha, (float, int)):
            alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        elif isinstance(self.alpha, (list, tuple, torch.Tensor)):
            alpha_values = torch.as_tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
            alpha_t = alpha_values[targets]
        else:
            alpha_t = 1.0

        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss
        assert torch.isfinite(focal_loss).all(), "[!] Numerical Error: FocalLoss contains NaN or Inf values!"

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
