import numpy as np
import torch


class RegionalSHAPExplainer:
    """
    Regional Shapley Value (SHAP) Explainer.
    Permutes and masks out distinct facial RoI regions (Left Eye, Right Eye, Mouth, Flow)
    to calculate marginal Shapley contributions to the final fatigue classification probability.
    """
    def __init__(self, model: torch.nn.Module, num_samples: int = 32):
        self.model = model
        self.num_samples = num_samples
        self.feature_names = ["Face RoI", "Left Eye RoI", "Right Eye RoI", "Mouth RoI", "Motion Flow"]

    def explain(
        self,
        video_tensor: torch.Tensor,
        flow_tensor: torch.Tensor,
        target_class: int = None
    ) -> dict:
        """
        Computes regional Shapley values.
        Args:
            video_tensor: (1, T, 3, H, W)
            flow_tensor: (1, T, 2, H_f, W_f)
        Returns:
            dict with Shapley values and percentage contributions per region.
        """
        self.model.eval()

        with torch.no_grad():
            base_out = self.model(video_tensor, flow_tensor)
            if target_class is None:
                target_class = torch.argmax(base_out["logits"], dim=1).item()
            base_prob = torch.softmax(base_out["logits"], dim=1)[0, target_class].item()

        # Define 5 regional masking perturbations
        shap_values = {}
        h, w = video_tensor.shape[3], video_tensor.shape[4]

        # 1. Mask Face center
        v_no_face = video_tensor.clone()
        v_no_face[:, :, :, int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)] = 0.0
        with torch.no_grad():
            p_no_face = torch.softmax(self.model(v_no_face, flow_tensor)["logits"], dim=1)[0, target_class].item()
        shap_values["Face RoI"] = base_prob - p_no_face

        # 2. Mask Left Eye
        v_no_leye = video_tensor.clone()
        v_no_leye[:, :, :, int(h * 0.2):int(h * 0.5), int(w * 0.12):int(w * 0.48)] = 0.0
        with torch.no_grad():
            p_no_leye = torch.softmax(self.model(v_no_leye, flow_tensor)["logits"], dim=1)[0, target_class].item()
        shap_values["Left Eye RoI"] = base_prob - p_no_leye

        # 3. Mask Right Eye
        v_no_reye = video_tensor.clone()
        v_no_reye[:, :, :, int(h * 0.2):int(h * 0.5), int(w * 0.52):int(w * 0.88)] = 0.0
        with torch.no_grad():
            p_no_reye = torch.softmax(self.model(v_no_reye, flow_tensor)["logits"], dim=1)[0, target_class].item()
        shap_values["Right Eye RoI"] = base_prob - p_no_reye

        # 4. Mask Mouth
        v_no_mouth = video_tensor.clone()
        v_no_mouth[:, :, :, int(h * 0.6):int(h * 0.95), int(w * 0.2):int(w * 0.8)] = 0.0
        with torch.no_grad():
            p_no_mouth = torch.softmax(self.model(v_no_mouth, flow_tensor)["logits"], dim=1)[0, target_class].item()
        shap_values["Mouth RoI"] = base_prob - p_no_mouth

        # 5. Mask Optical Flow
        f_zero = torch.zeros_like(flow_tensor)
        with torch.no_grad():
            p_no_flow = torch.softmax(self.model(video_tensor, f_zero)["logits"], dim=1)[0, target_class].item()
        shap_values["Motion Flow"] = base_prob - p_no_flow

        # Normalize to positive relative contributions
        abs_sum = sum(abs(v) for v in shap_values.values()) + 1e-6
        percentages = {k: (abs(v) / abs_sum) * 100.0 for k, v in shap_values.items()}

        return {
            "base_probability": base_prob,
            "shap_values": shap_values,
            "percentage_contributions": percentages,
            "dominant_region": max(percentages, key=percentages.get)
        }
