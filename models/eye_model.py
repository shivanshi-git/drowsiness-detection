import torch
import torch.nn as nn
from torchvision import models

class EyeStateModel(nn.Module):
    """
    Eye State Model: Specialized ResNet-18 trained exclusively on eye crops (128x128)
    to classify Eye Open vs Eye Closed (p_eye_closed).
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(EyeStateModel, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
        self.target_layer = self.backbone.layer4[-1]

    def forward(self, x):
        return self.backbone(x)

    def get_eye_closed_prob(self, x):
        """Returns softmax probability of eye closure."""
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1] # Probability of closed eye
