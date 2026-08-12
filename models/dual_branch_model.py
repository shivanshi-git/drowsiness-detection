import torch
import torch.nn as nn
from torchvision import models

class DualBranchResNet(nn.Module):
    """
    Two-Branch Multi-Modal Feature Fusion Architecture for Drowsiness Detection:
    - Eye Branch (128x128 crop)   -> ResNet Feature Extractor
    - Mouth Branch (128x128 crop) -> ResNet Feature Extractor
    - Fusion Layer -> Concatenation -> Dense Classifier Head
    
    Provides explicit feature disentanglement compared to passive side-by-side concatenation.
    Input Shape:
      - x_eye:   (B, 3, 128, 128)
      - x_mouth: (B, 3, 128, 128)
      - OR single dual-ROI image x_composite (B, 3, 128, 256) split into left and right halves.
    """
    def __init__(self, backbone_name='resnet18', num_classes=2, pretrained=True):
        super(DualBranchResNet, self).__init__()
        
        # 1. Eye Spatial Feature Branch
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.eye_branch = models.resnet18(weights=weights)
        in_feat_eye = self.eye_branch.fc.in_features
        self.eye_branch.fc = nn.Identity()

        # 2. Mouth Spatial Feature Branch
        self.mouth_branch = models.resnet18(weights=weights)
        in_feat_mouth = self.mouth_branch.fc.in_features
        self.mouth_branch.fc = nn.Identity()

        # 3. Fusion & Classifier Head
        total_features = in_feat_eye + in_feat_mouth
        self.fusion_classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_eye, x_mouth=None):
        """
        Accepts either separate x_eye and x_mouth tensors,
        OR a single combined x_composite tensor of shape (B, 3, 128, 256) split into left/right halves.
        """
        if x_mouth is None and x_eye.dim() == 4 and x_eye.shape[3] == 256:
            # Split composite (B, C, 128, 256) into left (eye) and right (mouth) halves
            x_composite = x_eye
            x_eye = x_composite[:, :, :, :128]
            x_mouth = x_composite[:, :, :, 128:]

        feat_eye = self.eye_branch(x_eye)     # (B, 512)
        feat_mouth = self.mouth_branch(x_mouth) # (B, 512)

        # Concatenate multi-modal spatial features
        fused_features = torch.cat([feat_eye, feat_mouth], dim=1) # (B, 1024)

        # Final classification
        logits = self.fusion_classifier(fused_features)
        return logits

if __name__ == "__main__":
    model = DualBranchResNet(backbone_name='resnet18')
    dummy_eye = torch.randn(2, 3, 128, 128)
    dummy_mouth = torch.randn(2, 3, 128, 128)
    out1 = model(dummy_eye, dummy_mouth)
    print(f"[✓] DualBranchResNet output shape (two inputs): {out1.shape}")

    dummy_composite = torch.randn(2, 3, 128, 256)
    out2 = model(dummy_composite)
    print(f"[✓] DualBranchResNet output shape (composite input): {out2.shape}")
