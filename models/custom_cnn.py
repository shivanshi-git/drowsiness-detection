import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """
    Residual Block with Conv2D, BatchNorm, and Skip Connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class CustomCNN(nn.Module):
    """
    Enhanced Deep Residual Custom CNN Architecture for Drowsiness Detection.
    Features 4-stage Residual Blocks with Skip Connections, Batch Normalization,
    and Dropout Regularization for high precision and fast convergence.
    """
    def __init__(self, num_classes=2, in_channels=3):
        super(CustomCNN, self).__init__()

        self.prep = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = ResBlock(32, 64, stride=2)   # 64x64
        self.layer2 = ResBlock(64, 128, stride=2)  # 32x32
        self.layer3 = ResBlock(128, 256, stride=2) # 16x16
        self.layer4 = ResBlock(256, 512, stride=2) # 8x8

        # Sequential container for easy Grad-CAM layer indexing
        self.features = nn.Sequential(
            self.prep,     # [0]
            self.layer1,   # [1]
            self.layer2,   # [2]
            self.layer3,   # [3]
            self.layer4,   # [4] Target Layer for Grad-CAM
            nn.AdaptiveAvgPool2d((1, 1)) # [5]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
