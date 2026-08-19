import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionModule(nn.Module):
    """
    Bidirectional Cross-Attention Layer between Spatial Region Tokens and Motion Flow Tokens.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_spatial = nn.LayerNorm(embed_dim)
        self.norm_flow = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_out = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, spatial_tokens: torch.Tensor, flow_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_tokens: (B, N_s, D)
            flow_tokens: (B, N_f, D)
        Returns:
            fused tokens: (B, N_s, D)
        """
        q = self.norm_spatial(spatial_tokens)
        kv = self.norm_flow(flow_tokens)

        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        x = spatial_tokens + attn_out
        x = x + self.mlp(self.norm_out(x))
        return x


class SpatialChannelAttention(nn.Module):
    """
    Spatial & Channel Attention (CBAM-inspired) for 1D token sequences / feature vectors.
    """
    def __init__(self, embed_dim: int = 256, reduction_ratio: int = 16):
        super().__init__()
        # Channel Attention
        reduced_dim = max(8, embed_dim // reduction_ratio)
        self.channel_gate = nn.Sequential(
            nn.Linear(embed_dim, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, embed_dim),
            nn.Sigmoid()
        )

        # Spatial / Sequence Token Attention
        self.token_gate = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, D)
        Returns:
            weighted token tensor: (B, N, D)
        """
        # Channel gating
        channel_weights = self.channel_gate(tokens.mean(dim=1, keepdim=True))  # (B, 1, D)
        tokens = tokens * channel_weights

        # Token / Spatial gating
        spatial_weights = self.token_gate(tokens)  # (B, N, 1)
        tokens = tokens * spatial_weights

        return tokens


class MultimodalFusionEngine(nn.Module):
    """
    Complete Cross-Attention + Spatial/Channel Attention Fusion Engine.
    Transforms spatial & motion token streams into a single compact frame-level feature vector.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = CrossAttentionModule(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.sc_attention = SpatialChannelAttention(embed_dim=embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, spatial_tokens: torch.Tensor, flow_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_tokens: (B, N_s, D)
            flow_tokens: (B, N_f, D)
        Returns:
            frame_embedding: (B, D)
        """
        fused = self.cross_attn(spatial_tokens, flow_tokens)
        refined = self.sc_attention(fused)  # (B, N_s, D)

        # Global average pool across tokens -> (B, D)
        frame_emb = self.pool(refined.transpose(1, 2)).squeeze(-1)
        return frame_emb
