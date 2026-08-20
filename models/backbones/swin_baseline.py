import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights


class SwinTransformerBaseline(nn.Module):
    """
    Swin Transformer (Swin-Tiny) hierarchical shifted window baseline.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = False):
        super().__init__()
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        self.model = swin_t(weights=weights)
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            logits = self.model(x).view(b, t, -1).mean(dim=1)
        else:
            logits = self.model(x)

        return {"logits": logits, "fatigue_score": torch.softmax(logits, dim=-1)[:, -1:]}
