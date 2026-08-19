import numpy as np
import torch


class IntegratedGradientsExplainer:
    """
    Integrated Gradients (IG) Explainer for Multi-Modal Low-Light Video Sequences.
    Axiomatic attribution satisfying Completeness and Sensitivity compared against a dark baseline.
    """
    def __init__(self, model: torch.nn.Module, steps: int = 20):
        self.model = model
        self.steps = steps

    def attribute(
        self,
        video_tensor: torch.Tensor,
        flow_tensor: torch.Tensor,
        target_class: int = None,
        baseline_video: torch.Tensor = None,
        baseline_flow: torch.Tensor = None
    ) -> dict:
        """
        Computes Integrated Gradients attribution maps for video and optical flow streams.
        Args:
            video_tensor: (1, T, 3, H, W)
            flow_tensor: (1, T, 2, H_f, W_f)
            target_class: class index to explain
        Returns:
            dict containing:
              'video_attributions': np.ndarray of shape (T, H, W)
              'flow_attributions': np.ndarray of shape (T, H_f, W_f)
              'spatial_importance_score': float
              'motion_importance_score': float
        """
        self.model.eval()

        if baseline_video is None:
            baseline_video = torch.zeros_like(video_tensor) # Completely dark frame baseline
        if baseline_flow is None:
            baseline_flow = torch.zeros_like(flow_tensor)   # Zero motion baseline

        if target_class is None:
            with torch.no_grad():
                out = self.model(video_tensor, flow_tensor)
                target_class = torch.argmax(out["logits"], dim=1).item()

        # Generate interpolated path steps: alpha in [0, 1]
        alphas = torch.linspace(0.0, 1.0, self.steps, device=video_tensor.device)
        grad_video_accum = torch.zeros_like(video_tensor)
        grad_flow_accum = torch.zeros_like(flow_tensor)

        for alpha in alphas:
            interp_video = baseline_video + alpha * (video_tensor - baseline_video)
            interp_flow = baseline_flow + alpha * (flow_tensor - baseline_flow)

            interp_video.requires_grad_(True)
            interp_flow.requires_grad_(True)

            out = self.model(interp_video, interp_flow)
            score = out["logits"][0, target_class]

            self.model.zero_grad()
            score.backward(retain_graph=True)

            if interp_video.grad is not None:
                grad_video_accum += interp_video.grad
            if interp_flow.grad is not None:
                grad_flow_accum += interp_flow.grad

        # Average gradients * (input - baseline)
        avg_grad_video = grad_video_accum / float(self.steps)
        avg_grad_flow = grad_flow_accum / float(self.steps)

        ig_video = (video_tensor - baseline_video) * avg_grad_video
        ig_flow = (flow_tensor - baseline_flow) * avg_grad_flow

        # Aggregate across color/flow channels: (1, T, C, H, W) -> (T, H, W)
        attr_video = ig_video.abs().mean(dim=2).squeeze(0).detach().cpu().numpy()
        attr_flow = ig_flow.abs().mean(dim=2).squeeze(0).detach().cpu().numpy()

        # Importance percentages
        v_sum = float(attr_video.sum())
        f_sum = float(attr_flow.sum())
        total = max(1e-5, v_sum + f_sum)

        return {
            "video_attributions": attr_video,
            "flow_attributions": attr_flow,
            "spatial_importance_pct": (v_sum / total) * 100.0,
            "motion_importance_pct": (f_sum / total) * 100.0
        }
