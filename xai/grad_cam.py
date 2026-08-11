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


class LayerCAM:
    """
    LayerCAM: Fine-grained spatial attribution maps on 128x128 eye crops
    by computing element-wise spatial gradient-activation products.
    Eliminates 4x4 spatial grid degradation.
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
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        score = output[0, class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32), class_idx, F.softmax(output, dim=1)[0, class_idx].item()

        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()

        # Element-wise positive spatial weighting (LayerCAM)
        positive_grads = np.maximum(gradients, 0)
        cam = np.sum(positive_grads * activations, axis=0)
        cam = np.maximum(cam, 0)

        if cam.max() > 0:
            cam = cam / cam.max()

        _, _, h, w = input_tensor.shape
        cam_resized = cv2.resize(cam, (w, h))
        confidence = F.softmax(output, dim=1)[0, class_idx].item()

        return cam_resized, class_idx, confidence

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


class IntegratedGradients:
    """
    Model-Agnostic Integrated Gradients (IG).
    Integrates gradients along straight path from baseline to input crop.
    Enables unified cross-paradigm explanation for both CNNs and Vision Transformers (ViT).
    """
    def __init__(self, model):
        self.model = model

    def generate_heatmap(self, input_tensor, class_idx=None, steps=20):
        self.model.eval()

        if class_idx is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                class_idx = torch.argmax(output, dim=1).item()

        baseline = torch.zeros_like(input_tensor)
        scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]

        grads = []
        for scaled_input in scaled_inputs:
            scaled_input = scaled_input.clone().detach().requires_grad_(True)
            output = self.model(scaled_input)
            score = output[0, class_idx]
            self.model.zero_grad()
            score.backward()
            grads.append(scaled_input.grad.detach().cpu().numpy()[0])

        avg_grads = np.mean(np.array(grads), axis=0)
        delta = (input_tensor - baseline).cpu().numpy()[0]
        integrated_grad = delta * avg_grads

        attribution = np.sum(np.abs(integrated_grad), axis=0)
        attribution = np.maximum(attribution, 0)

        if attribution.max() > 0:
            attribution = attribution / attribution.max()

        with torch.no_grad():
            output = self.model(input_tensor)
            confidence = F.softmax(output, dim=1)[0, class_idx].item()

        return attribution, class_idx, confidence
