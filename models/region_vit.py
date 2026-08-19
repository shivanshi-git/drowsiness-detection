import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Splits image into non-overlapping patches and projects them to embed_dim.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, embed_dim, grid, grid) -> (B, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ViTBlock(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class RegionAwareViT(nn.Module):
    """
    Region-Aware Vision Transformer.
    Integrates patch embeddings from Face, Left Eye, Right Eye, and Mouth with region-type embeddings.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Patch Embedders for different RoI scales
        self.face_embed = PatchEmbedding(img_size=224, patch_size=16, embed_dim=embed_dim)   # 14x14 = 196 patches
        self.eye_embed = PatchEmbedding(img_size=64, patch_size=16, embed_dim=embed_dim)     # 4x4 = 16 patches
        self.mouth_embed = PatchEmbedding(img_size=64, patch_size=16, embed_dim=embed_dim)   # 4x4 = 16 patches

        # Learnable Region Type Embeddings (0: Face, 1: Left Eye, 2: Right Eye, 3: Mouth)
        self.region_type_embed = nn.Embedding(4, embed_dim)

        # Class Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional Encoding projection
        total_patches = 196 + 16 + 16 + 16 + 1  # 245 tokens
        self.pos_embed = nn.Parameter(torch.randn(1, total_patches, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        face: torch.Tensor,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        mouth: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            face: (B, 3, 224, 224)
            left_eye: (B, 3, 64, 64)
            right_eye: (B, 3, 64, 64)
            mouth: (B, 3, 64, 64)
        Returns:
            token sequence tensor of shape (B, num_tokens, embed_dim)
        """
        b = face.shape[0]

        # 1. Extract and tag patch tokens
        p_face = self.face_embed(face) + self.region_type_embed(torch.tensor(0, device=face.device))
        p_leye = self.eye_embed(left_eye) + self.region_type_embed(torch.tensor(1, device=face.device))
        p_reye = self.eye_embed(right_eye) + self.region_type_embed(torch.tensor(2, device=face.device))
        p_mouth = self.mouth_embed(mouth) + self.region_type_embed(torch.tensor(3, device=face.device))

        # 2. Prepend CLS token and concatenate
        cls_tokens = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls_tokens, p_face, p_leye, p_reye, p_mouth], dim=1)  # (B, 245, embed_dim)

        # 3. Add positional embeddings
        tokens = self.pos_drop(tokens + self.pos_embed[:, :tokens.shape[1], :])

        # 4. Forward through Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        return tokens
