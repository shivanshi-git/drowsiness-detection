import torch
import torch.nn as nn
from torchvision import models

class TemporalDrowsinessModel(nn.Module):
    """
    Temporal Drowsiness Architecture combining Spatial Feature Extractor (ResNet-18)
    with a Temporal GRU Sequence Aggregator for 10-frame video sequence modeling.
    
    Resolves the 0.2s natural blink vs. >1.5s micro-sleep classification ambiguity.
    Input Shape: (batch_size, seq_len, channels, height, width) -> (B, 10, 3, 128, 128)
    """
    def __init__(self, backbone_name='resnet18', hidden_dim=128, num_classes=2, pretrained=True):
        super(TemporalDrowsinessModel, self).__init__()
        self.backbone_name = backbone_name
        self.seq_len = 10
        
        # Load spatial backbone
        if backbone_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity() # Remove default FC layer
        else:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        # Recurrent Sequence Aggregator (GRU)
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_seq):
        """
        x_seq: Tensor of shape (B, seq_len, C, H, W)
        Returns: Logits tensor of shape (B, num_classes)
        """
        if x_seq.dim() == 4:
            # Single frame tensor fallback (B, C, H, W) -> unsqueeze sequence dimension
            x_seq = x_seq.unsqueeze(1)

        b, seq_len, c, h, w = x_seq.shape
        
        # Flatten (B, seq_len, C, H, W) into (B * seq_len, C, H, W) for batch spatial feature extraction
        x_flat = x_seq.reshape(b * seq_len, c, h, w)
        features = self.backbone(x_flat) # (B * seq_len, in_features)

        # Reshape back to sequence format (B, seq_len, in_features)
        features_seq = features.reshape(b, seq_len, -1)

        # Pass through temporal GRU
        gru_out, _ = self.gru(features_seq) # (B, seq_len, hidden_dim)

        # Extract last frame representation
        final_temporal_state = gru_out[:, -1, :] # (B, hidden_dim)

        # Compute output classification logits
        logits = self.classifier(final_temporal_state)
        return logits

if __name__ == "__main__":
    model = TemporalDrowsinessModel(backbone_name='resnet18', hidden_dim=128)
    dummy_seq = torch.randn(2, 10, 3, 128, 128)
    output = model(dummy_seq)
    print(f"[✓] TemporalDrowsinessModel test output shape: {output.shape}")
