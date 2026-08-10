import torch
import torch.nn as nn
from torchvision import models

class ViTTinyModel(nn.Module):
    """
    Vision Transformer (ViT-B-16 / ViT-Tiny) for Attention-based Drowsiness Detection.
    """
    def __init__(self, model_name='vit_tiny', num_classes=2, pretrained=True):
        super(ViTTinyModel, self).__init__()

        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vit_b_16(weights=weights)
        
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # Resize input dynamically to 224x224 if necessary for ViT patch encoding
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.backbone(x)
