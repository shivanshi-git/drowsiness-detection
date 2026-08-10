import torch
import torch.nn as nn
from torchvision import models

class VGGModel(nn.Module):
    """
    VGG16 / VGG19 Transfer Learning Model with ImageNet pre-trained backbone.
    Target layer for Grad-CAM: features[28] (block5_conv3 in VGG16).
    """
    def __init__(self, model_name='vgg16', num_classes=2, pretrained=True):
        super(VGGModel, self).__init__()
        
        weights = models.VGG16_Weights.DEFAULT if (pretrained and model_name == 'vgg16') else None
        if model_name == 'vgg19':
            weights = models.VGG19_Weights.DEFAULT if pretrained else None
            backbone = models.vgg19(weights=weights)
        else:
            backbone = models.vgg16(weights=weights)

        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Replace classifier head for binary/multi-class drowsiness detection
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
