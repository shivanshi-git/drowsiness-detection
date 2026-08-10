import torch
import torch.nn as nn
from torchvision import models

class EfficientNetModel(nn.Module):
    """
    EfficientNet-B0 / EfficientNet-B2 Model.
    Target layer for Grad-CAM: backbone.features[-1].
    """
    def __init__(self, model_name='efficientnet_b0', num_classes=2, pretrained=True):
        super(EfficientNetModel, self).__init__()

        if model_name == 'efficientnet_b2':
            weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b2(weights=weights)
        else:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
