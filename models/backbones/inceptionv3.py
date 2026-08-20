import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights


class InceptionV3Baseline(nn.Module):
    """
    Inception-v3 baseline for multi-scale spatial drowsiness feature benchmark.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = False):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT if pretrained else None
        self.model = inception_v3(weights=weights, aux_logits=False, transform_input=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            if (h, w) != (299, 299):
                x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear')
            logits = self.model(x).view(b, t, -1).mean(dim=1)
        else:
            logits = self.model(x)

        return {"logits": logits, "fatigue_score": torch.softmax(logits, dim=-1)[:, -1:]}
