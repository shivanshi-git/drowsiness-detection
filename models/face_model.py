import torch
import torch.nn as nn
from torchvision import models

class FaceDrowsinessModel(nn.Module):
    """
    Face Drowsiness Model: Operates on 224x224 Full Face Crops (preserving eyes, mouth,
    nose, and head posture) to classify ALERT, DROWSY, and YAWNING states.
    """
    def __init__(self, num_classes=3, pretrained=True):
        super(FaceDrowsinessModel, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        self.target_layer = self.backbone.layer4[-1]

    def forward(self, x):
        return self.backbone(x)

    def get_face_probs(self, x):
        """Returns softmax probabilities across [ALERT, DROWSY, YAWNING]."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
