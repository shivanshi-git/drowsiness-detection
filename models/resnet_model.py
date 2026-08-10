import torch
import torch.nn as nn
from torchvision import models

class ResNetModel(nn.Module):
    """
    ResNet18 / ResNet50 Transfer Learning Model.
    Target layer for Grad-CAM: backbone.layer4.
    """
    def __init__(self, model_name='resnet18', num_classes=2, pretrained=True):
        super(ResNetModel, self).__init__()

        if model_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
        else:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features

        # Replace classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
