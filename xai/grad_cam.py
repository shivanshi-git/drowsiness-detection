import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Computes fine-grained activation heatmaps highlighting regions of the face/eye/mouth
    that drive the driver drowsiness prediction.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generates a 2D Grad-CAM heatmap array normalized between 0 and 1.
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        score = output[0, class_idx]
        
        # Backward pass
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if hooks were skipped
            return np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32), class_idx, F.softmax(output, dim=1)[0, class_idx].item()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        # Global Average Pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))

        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU to retain positive attributions
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize heatmap to input tensor dimensions
        _, _, h, w = input_tensor.shape
        cam_resized = cv2.resize(cam, (w, h))

        probs = F.softmax(output, dim=1)[0]
        confidence = probs[class_idx].item()

        return cam_resized, class_idx, confidence

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
