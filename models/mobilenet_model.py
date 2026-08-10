import torch
import torch.nn as nn
from torchvision import models

class MobileNetModel(nn.Module):
    """
    MobileNetV2 / MobileNetV3 Lightweight Model for Real-Time Edge Processing.
    Target layer for Grad-CAM: backbone.features[-1].
    """
    def __init__(self, model_name='mobilenet_v2', num_classes=2, pretrained=True):
        super(MobileNetModel, self).__init__()

        if model_name == 'mobilenet_v3':
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1024, num_classes),
            )
        else:
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, num_classes)
            )

    def forward(self, x):
        return self.backbone(x)
