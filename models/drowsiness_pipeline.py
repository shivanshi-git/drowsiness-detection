import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .llformer import LLFormer
from .region_vit import RegionAwareViT
from .flow_vit import OpticalFlowViT
from .cross_attention_fusion import MultimodalFusionEngine
from .temporal_transformer import TemporalSequenceTransformer
from .retinaface_detector import RetinaFaceDetector


class LowLightDrowsinessPipeline(nn.Module):
    """
    End-to-End State-of-the-Art Low-Light Drowsiness Detection Architecture.
    
    Data Flow:
      Raw Low-Light Sequence (B, T, 3, H, W) + Flow (B, T, 2, H, W)
      └── LLFormer Low-Light Restoration
          └── RoI Decomposition (Face, Left Eye, Right Eye, Mouth)
              ├── Region-Aware ViT (Spatial Stream)
              └── Optical Flow ViT (Motion Stream)
                  └── Cross-Attention Fusion
                      └── Spatial & Channel Attention
                          └── Temporal Sequence Transformer
                              └── Multi-Class Drowsiness & Fatigue Prediction
    """
    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 256,
        sequence_length: int = 16,
        enable_llformer: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        self.enable_llformer = enable_llformer

        # 1. Low-Light Enhancement Transformer
        if enable_llformer:
            self.llformer = LLFormer(in_channels=3, out_channels=3, dim=32, num_blocks=2)

        # 2. Spatial Stream: Region-Aware ViT
        self.region_vit = RegionAwareViT(
            embed_dim=embed_dim,
            depth=4,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.1
        )

        # 3. Motion Stream: Optical Flow ViT
        self.flow_vit = OpticalFlowViT(
            img_size=112,
            patch_size=16,
            in_channels=2,
            embed_dim=embed_dim,
            depth=4,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.1
        )

        # 4. Multi-modal Fusion: Cross-Attention + Spatial/Channel Attention
        self.fusion = MultimodalFusionEngine(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1
        )

        # 5. Temporal Sequence Modeling
        self.temporal_transformer = TemporalSequenceTransformer(
            embed_dim=embed_dim,
            max_seq_len=64,
            depth=4,
            num_heads=8,
            mlp_ratio=4.0,
            num_classes=num_classes,
            dropout=0.1
        )

        # RetinaFace detector for preprocessing / RoI extraction during inference
        self.detector = RetinaFaceDetector()

    def _extract_rois_from_tensor(self, frame_tensor: torch.Tensor) -> tuple:
        """
        Differentiable / geometric RoI slicing from (B, 3, H, W) normalized tensor.
        Extracts Face, Left Eye, Right Eye, and Mouth.
        """
        b, c, h, w = frame_tensor.shape
        
        # Face: center crop (224, 224)
        face = F.interpolate(frame_tensor, size=(224, 224), mode='bilinear', align_corners=False)

        # Left eye: top-left quadrant (64, 64)
        le_y1, le_y2 = int(h * 0.20), int(h * 0.50)
        le_x1, le_x2 = int(w * 0.12), int(w * 0.48)
        left_eye = frame_tensor[:, :, le_y1:le_y2, le_x1:le_x2]
        left_eye = F.interpolate(left_eye, size=(64, 64), mode='bilinear', align_corners=False)

        # Right eye: top-right quadrant (64, 64)
        re_y1, re_y2 = int(h * 0.20), int(h * 0.50)
        re_x1, re_x2 = int(w * 0.52), int(w * 0.88)
        right_eye = frame_tensor[:, :, re_y1:re_y2, re_x1:re_x2]
        right_eye = F.interpolate(right_eye, size=(64, 64), mode='bilinear', align_corners=False)

        # Mouth: bottom center (64, 64)
        m_y1, m_y2 = int(h * 0.60), int(h * 0.95)
        m_x1, m_x2 = int(w * 0.20), int(w * 0.80)
        mouth = frame_tensor[:, :, m_y1:m_y2, m_x1:m_x2]
        mouth = F.interpolate(mouth, size=(64, 64), mode='bilinear', align_corners=False)

        return face, left_eye, right_eye, mouth

    def forward(self, video: torch.Tensor, flow: torch.Tensor) -> dict:
        """
        Args:
            video: (B, T, 3, H, W) raw low-light sequence
            flow: (B, T, 2, H_f, W_f) optical flow sequence
        Returns:
            dict containing:
              'logits': (B, num_classes)
              'fatigue_score': (B, 1)
              'enhanced_video': (B, T, 3, H, W)
        """
        b, t, c, h, w = video.shape
        _, _, _, h_f, w_f = flow.shape

        # Flatten batch and time dimensions for frame-level processing
        flat_video = video.view(b * t, c, h, w)
        flat_flow = flow.view(b * t, 2, h_f, w_f)

        # 1. LLFormer Low-Light Enhancement
        if self.enable_llformer:
            enhanced_video = self.llformer(flat_video)
        else:
            enhanced_video = flat_video

        # 2. Extract RoIs for Region-Aware ViT
        face, leye, reye, mouth = self._extract_rois_from_tensor(enhanced_video)

        # 3. Spatial Tokens
        spatial_tokens = self.region_vit(face, leye, reye, mouth)  # (B*T, N_s, D)

        # 4. Motion Flow Tokens
        flow_tokens = self.flow_vit(flat_flow)                      # (B*T, N_f, D)

        # 5. Multimodal Cross-Attention & Spatial/Channel Attention Fusion
        frame_embeddings = self.fusion(spatial_tokens, flow_tokens)  # (B*T, D)

        # 6. Reshape back to sequence: (B, T, D)
        seq_embeddings = frame_embeddings.view(b, t, self.embed_dim)

        # 7. Temporal Transformer Sequence Modeling
        out = self.temporal_transformer(seq_embeddings)
        out["enhanced_video"] = enhanced_video.view(b, t, c, h, w)

        return out
