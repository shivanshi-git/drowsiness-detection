import torch
import torch.nn as nn
import torch.nn.functional as F


class MDTA(nn.Module):
    """
    Multi-Dconv Head Transposed Attention (MDTA).
    Calculates cross-covariance across channels rather than spatial dimensions,
    allowing efficient global attention on high-resolution low-light frames.
    """
    def __init__(self, channels: int, num_heads: int = 4, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3, bias=bias)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)
        return out


class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network.
    Controls information flow through gating mechanism to remove low-light noise.
    """
    def __init__(self, channels: int, ffn_expansion_factor: float = 2.66, bias: bool = False):
        super().__init__()
        hidden_features = int(channels * ffn_expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, ffn_expansion_factor: float = 2.66, bias: bool = False):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = MDTA(dim, num_heads=num_heads, bias=bias)
        self.norm2 = nn.GroupNorm(1, dim)
        self.ffn = GDFN(dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LLFormer(nn.Module):
    """
    Low-Light Enhancement Transformer (LLFormer).
    Restores degraded contrast, illuminates underexposed regions, and suppresses dark-channel noise.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, dim: int = 32, num_blocks: int = 2):
        super().__init__()
        self.embedding = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=4) for _ in range(num_blocks)
        ])
        self.refinement = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) or (B*T, 3, H, W) low-light image tensor in [0, 1]
        Returns:
            enhanced image tensor of same shape
        """
        residual = x
        feat = self.embedding(x)
        for block in self.blocks:
            feat = block(feat)
        enhanced = self.refinement(feat) + residual
        return torch.sigmoid(enhanced)
