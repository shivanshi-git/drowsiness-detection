import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAMExplainer:
    """
    Grad-CAM & ViT Attention Map Generator for Region-Aware Vision Transformer & LLFormer.
    Produces high-resolution visual heatmaps overlaying input low-light frames.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module = None):
        self.model = model
        self.target_layer = target_layer or model.region_vit.blocks[-1].norm1
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        video_tensor: torch.Tensor,
        flow_tensor: torch.Tensor,
        target_class: int = None
    ) -> np.ndarray:
        """
        Generates 2D Grad-CAM heatmap over the video sequence.
        Args:
            video_tensor: (1, T, 3, H, W)
            flow_tensor: (1, T, 2, H_f, W_f)
            target_class: Integer class index to explain
        Returns:
            np.ndarray of shape (H, W) heatmap in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        out = self.model(video_tensor, flow_tensor)
        logits = out["logits"]

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback uniform heatmap
            return np.ones((video_tensor.shape[3], video_tensor.shape[4]), dtype=np.float32)

        # Activations shape: (T, num_tokens, D)
        # Take sequence average
        act = self.activations  # (B*T, N, D)
        grad = self.gradients   # (B*T, N, D)

        # Token importance weights
        weights = torch.mean(grad, dim=-1, keepdim=True) # (B*T, N, 1)
        cam = torch.sum(weights * act, dim=-1)           # (B*T, N)
        cam = F.relu(cam)

        # Extract face tokens (first 196 tokens after CLS token)
        face_tokens = cam[:, 1:197].mean(dim=0)          # (196,)
        grid_size = int(np.sqrt(face_tokens.shape[0]))   # 14x14
        heatmap_2d = face_tokens.view(grid_size, grid_size).detach().cpu().numpy()

        # Normalize and upscale to image resolution (H, W)
        heatmap_2d = (heatmap_2d - np.min(heatmap_2d)) / max(1e-5, (np.max(heatmap_2d) - np.min(heatmap_2d)))
        heatmap_resized = cv2.resize(heatmap_2d, (video_tensor.shape[4], video_tensor.shape[3]))
        return heatmap_resized

    @staticmethod
    def overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Overlays Jet colormap heatmap onto original image frame.
        """
        heatmap_uint8 = np.uint8(255 * heatmap)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(color_heatmap, alpha, image_bgr, 1.0 - alpha, 0)
        return overlay
