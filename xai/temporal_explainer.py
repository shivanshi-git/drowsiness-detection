import numpy as np
import torch


class TemporalAttentionExplainer:
    """
    Temporal Behavior & Confidence Trajectory Explainer.
    Answers WHEN the model became confident that the driver was drowsy over a temporal sequence window.
    
    Generates:
      - Frame-by-frame confidence progression: P(Drowsy) over 0s -> 5s timeline
      - Confidence onset transition point (when fatigue crossed critical threshold)
      - Natural-language temporal progression narrative
    """
    def __init__(self, model: torch.nn.Module, fps: float = 3.0):
        self.model = model
        self.fps = fps
        self.temporal_attn_weights = None
        self._register_hook()

    def _register_hook(self):
        last_block = self.model.temporal_transformer.blocks[-1]
        
        def hook(module, input, output):
            norm_x = module.norm1(input[0])
            _, weights = module.attn(norm_x, norm_x, norm_x, need_weights=True)
            self.temporal_attn_weights = weights

        last_block.register_forward_hook(hook)

    def explain_temporal_behavior(
        self,
        video_tensor: torch.Tensor,
        flow_tensor: torch.Tensor,
        alert_threshold: float = 0.60
    ) -> dict:
        """
        Args:
            video_tensor: (1, T, 3, H, W)
            flow_tensor: (1, T, 2, H_f, W_f)
            alert_threshold: probability threshold considered drowsy
        Returns:
            dict containing timeline timestamps, frame-by-frame probabilities,
            confidence curve, transition onset time, and narrative explanation.
        """
        self.model.eval()
        t_len = video_tensor.shape[1]
        total_duration_sec = float(t_len / self.fps)
        timeline_sec = np.linspace(0.0, total_duration_sec, t_len)

        # 1. Step-wise forward simulation across sub-windows to get exact P(Drowsy) over time
        frame_confidences = []
        with torch.no_grad():
            for t in range(1, t_len + 1):
                sub_video = video_tensor[:, :t]
                sub_flow = flow_tensor[:, :t]
                
                # Zero pad to standard sequence length
                if t < t_len:
                    pad_v = torch.zeros((1, t_len - t, video_tensor.shape[2], video_tensor.shape[3], video_tensor.shape[4]), device=video_tensor.device)
                    pad_f = torch.zeros((1, t_len - t, flow_tensor.shape[2], flow_tensor.shape[3], flow_tensor.shape[4]), device=flow_tensor.device)
                    v_in = torch.cat([sub_video, pad_v], dim=1)
                    f_in = torch.cat([sub_flow, pad_f], dim=1)
                else:
                    v_in, f_in = sub_video, sub_flow

                out = self.model(v_in, f_in)
                p_drowsy = out["fatigue_score"].item()
                frame_confidences.append(p_drowsy)

        # 2. Attention weights from transformer
        if self.temporal_attn_weights is not None:
            cls_attn = self.temporal_attn_weights[0, 0, 1:t_len + 1].detach().cpu().numpy()
            cls_attn = (cls_attn - np.min(cls_attn)) / max(1e-5, (np.max(cls_attn) - np.min(cls_attn)))
        else:
            cls_attn = np.array(frame_confidences)

        # 3. Detect transition onset point
        onset_idx = None
        for i, conf in enumerate(frame_confidences):
            if conf >= alert_threshold:
                onset_idx = i
                break

        onset_time_sec = float(timeline_sec[onset_idx]) if onset_idx is not None else None

        # 4. Generate Narrative Explanation
        initial_prob = frame_confidences[0]
        final_prob = frame_confidences[-1]
        delta_prob = final_prob - initial_prob

        if onset_idx is not None:
            narrative = (
                f"The model's drowsiness confidence rose steadily from {initial_prob*100:.0f}% at 0.0s "
                f"to {final_prob*100:.0f}% over a {total_duration_sec:.1f}s window, "
                f"crossing the fatigue threshold at t = {onset_time_sec:.1f}s."
            )
        elif final_prob > initial_prob:
            narrative = (
                f"Drowsiness confidence showed a mild upward trend ({initial_prob*100:.0f}% -> {final_prob*100:.0f}%) "
                f"over {total_duration_sec:.1f}s without exceeding the alert threshold."
            )
        else:
            narrative = (
                f"Driver alertness remained stable (mean confidence: {np.mean(frame_confidences)*100:.0f}%) "
                f"across the entire {total_duration_sec:.1f}s temporal window."
            )

        return {
            "timeline_seconds": [round(s, 2) for s in timeline_sec.tolist()],
            "drowsiness_probabilities": [round(p, 3) for p in frame_confidences],
            "attention_weights": [round(float(a), 3) for a in cls_attn.tolist()],
            "transition_onset_sec": onset_time_sec,
            "total_window_duration_sec": total_duration_sec,
            "narrative_explanation": narrative
        }
