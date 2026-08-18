import cv2
import torch
import numpy as np
from .grad_cam import GradCAM
from .visualizer import overlay_heatmap

class HierarchicalXAIVisualizer:
    """
    Hierarchical Multi-Region XAI Visualizer:
    Computes activation heatmaps over 224x224 full-face driver images,
    highlighting eye closure and yawning regions simultaneously.
    """
    def __init__(self, face_model, eye_model=None):
        self.face_model = face_model
        self.eye_model = eye_model

    def generate_evidence_overlay(self, face_bgr, tensor_224, alpha=0.5):
        """
        Generates Grad-CAM activation heatmap for face model and overlays on original face image.
        """
        if tensor_224 is None or self.face_model is None:
            return face_bgr, np.zeros_like(face_bgr)

        device = next(self.face_model.parameters()).device
        tensor_224 = tensor_224.to(device)

        # Grad-CAM on Face Model
        target_layer = getattr(self.face_model, 'target_layer', None)
        if target_layer is None and hasattr(self.face_model, 'backbone'):
            target_layer = self.face_model.backbone.layer4[-1]

        grad_cam = GradCAM(self.face_model, target_layer)
        heatmap, class_idx, conf = grad_cam.generate_heatmap(tensor_224)
        grad_cam.remove_hooks()

        blended_bgr, color_heatmap = overlay_heatmap(face_bgr, heatmap, alpha=alpha)
        return blended_bgr, color_heatmap
