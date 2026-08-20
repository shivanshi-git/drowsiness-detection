import torch
import torch.nn as nn
from .region_vit import ViTBlock


class TemporalSequenceTransformer(nn.Module):
    """
    Temporal Transformer Encoder.
    Models long-range temporal dependencies across video frame representations (e.g. 16/32 frames).
    Captures progressive fatigue transitions, PERCLOS trends, blink duration, and microsleeps.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        max_seq_len: int = 64,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Temporal CLS token representing the whole video sequence state
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Temporal Positional Encodings
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Multi-class Drowsiness Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # Binary Fatigue Head (Alert vs Drowsy) for fast alerting
        self.binary_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, seq_embeddings: torch.Tensor) -> dict:
        """
        Args:
            seq_embeddings: (B, T, D) sequence of frame embeddings
        Returns:
            dict containing:
              'logits': (B, num_classes)
              'fatigue_score': (B, 1) probability in [0, 1]
              'temporal_features': (B, D)
        """
        b, t, d = seq_embeddings.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, seq_embeddings], dim=1)  # (B, T+1, D)
        x = self.pos_drop(x + self.pos_embed[:, :t + 1, :])

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # Sequence CLS representation

        logits = self.classifier(cls_out)
        fatigue_score = self.binary_head(cls_out)

        return {
            "logits": logits,
            "fatigue_score": fatigue_score,
            "temporal_features": cls_out
        }
