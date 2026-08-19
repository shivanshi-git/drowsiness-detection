import torch
import torch.nn as nn
from .region_vit import PatchEmbedding, ViTBlock


class OpticalFlowViT(nn.Module):
    """
    Optical Flow Vision Transformer.
    Encodes dense 2-channel velocity fields (dx, dy) into fine-grained motion tokens.
    """
    def __init__(
        self,
        img_size: int = 112,
        patch_size: int = 16,
        in_channels: int = 2,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches  # (112/16)^2 = 49

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flow: (B, 2, H, W) optical flow tensor
        Returns:
            motion tokens tensor of shape (B, num_flow_tokens, embed_dim)
        """
        b = flow.shape[0]
        x = self.patch_embed(flow)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x
