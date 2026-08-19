import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Baseline(nn.Module):
    """
    Standard ResNet-50 2D baseline for frame-level drowsiness classification.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.model = resnet50(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        # If sequence (B, T, 3, H, W), average pool over T
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            logits = self.model(x).view(b, t, -1).mean(dim=1)
        else:
            logits = self.model(x)

        return {"logits": logits, "fatigue_score": torch.softmax(logits, dim=-1)[:, -1:]}
