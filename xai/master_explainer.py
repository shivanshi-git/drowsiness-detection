import os
import cv2
import numpy as np
import torch

from .grad_cam import GradCAMExplainer
from .integrated_gradients import IntegratedGradientsExplainer
from .shap_explainer import RegionalSHAPExplainer
from .temporal_explainer import TemporalAttentionExplainer
from .landmark_explainer import LandmarkExplainer
from .alarm_explainer import ExplainableAlarmReasoner


class MasterXAIExplainer:
    """
    Unified Master Explainability Suite.
    Integrates Grad-CAM, Integrated Gradients, Regional SHAP, Temporal Timeline Behavior,
    Facial Landmarks, and Explainable Alarm Card generation.
    """
    def __init__(self, model: torch.nn.Module, fps: float = 3.0):
        self.model = model
        self.grad_cam = GradCAMExplainer(model)
        self.integrated_grads = IntegratedGradientsExplainer(model, steps=12)
        self.shap_explainer = RegionalSHAPExplainer(model)
        self.temporal_explainer = TemporalAttentionExplainer(model, fps=fps)
        self.landmark_explainer = LandmarkExplainer()
        self.alarm_reasoner = ExplainableAlarmReasoner()

    def generate_full_explanation(
        self,
        video_tensor: torch.Tensor,
        flow_tensor: torch.Tensor,
        raw_last_frame_bgr: np.ndarray = None,
        target_class: int = None,
        perclos: float = 0.28,
        closure_duration: float = 2.4,
        alarm_level: int = 2
    ) -> dict:
        """
        Executes complete multi-modal XAI analysis.
        """
        # 1. Grad-CAM Spatial Heatmap
        cam_heatmap = self.grad_cam.generate_cam(video_tensor, flow_tensor, target_class=target_class)

        # 2. Regional SHAP Values
        shap_result = self.shap_explainer.explain(video_tensor, flow_tensor, target_class=target_class)

        # 3. Temporal Confidence Timeline (WHEN the model became confident)
        temporal_result = self.temporal_explainer.explain_temporal_behavior(video_tensor, flow_tensor)

        # 4. Landmark Geometry
        landmark_result = self.landmark_explainer.explain_landmarks({})

        # 5. Explainable Alarm Card (Reasoning behind the alert)
        final_prob = temporal_result["drowsiness_probabilities"][-1]
        alarm_card = self.alarm_reasoner.generate_alarm_card(
            drowsy_prob=final_prob,
            predicted_class=target_class or 4,
            ear_value=landmark_result["avg_ear"],
            mar_value=landmark_result["mar"],
            head_pitch=landmark_result["head_pitch"],
            perclos=perclos,
            closure_duration=closure_duration,
            temporal_window_sec=temporal_result["total_window_duration_sec"],
            alarm_level=alarm_level
        )

        # 6. Render High-Resolution Composite XAI Dashboard
        composite_bgr = None
        if raw_last_frame_bgr is not None:
            composite_bgr = self.create_xai_dashboard_canvas(
                raw_last_frame_bgr,
                cam_heatmap,
                shap_result["percentage_contributions"],
                temporal_result,
                alarm_card
            )

        return {
            "grad_cam_heatmap": cam_heatmap,
            "shap_attribution": shap_result,
            "temporal_behavior": temporal_result,
            "landmark_explanation": landmark_result,
            "alarm_card": alarm_card,
            "composite_image": composite_bgr
        }

    def create_xai_dashboard_canvas(
        self,
        frame_bgr: np.ndarray,
        heatmap: np.ndarray,
        shap_percentages: dict,
        temporal_data: dict,
        alarm_card: dict
    ) -> np.ndarray:
        """
        Creates a dual-view monitor:
          Left: Grad-CAM Saliency Overlay + Real-time bounding
          Right: Comprehensive XAI Diagnosis Panel (Temporal Graph + Alarm Reason Card + SHAP)
        """
        h, w = frame_bgr.shape[:2]
        panel_w = max(420, w)
        canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)

        # 1. Left Panel: Grad-CAM Overlay
        cam_overlay = self.grad_cam.overlay_heatmap(frame_bgr, heatmap, alpha=0.55)
        canvas[:, :w] = cam_overlay
        cv2.putText(canvas, "WHERE THE MODEL LOOKED (Grad-CAM)", (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 255), 1)

        # 2. Right Panel: Diagnosis & Temporal Explanation
        right = np.full((h, panel_w, 3), 24, dtype=np.uint8)
        y = 25

        # Section A: Explainable Alarm Card
        level = alarm_card["alarm_level"]
        card_color = (0, 0, 255) if level == 3 else ((0, 140, 255) if level == 2 else (0, 255, 255))
        title = "CRITICAL ALARM" if level == 3 else ("DROWSINESS ALERT" if level == 2 else "CAUTION NOTICE")
        
        cv2.rectangle(right, (10, y - 15), (panel_w - 10, y + 105), (40, 40, 45), -1)
        cv2.rectangle(right, (10, y - 15), (panel_w - 10, y + 105), card_color, 2)
        cv2.putText(right, f"[ {title} ] - {alarm_card['drowsiness_prob_pct']:.1f}% Confidence", (20, y + 8), cv2.FONT_HERSHEY_DUPLEX, 0.55, card_color, 1)
        
        cv2.putText(right, f"Eye Closure: {alarm_card['eye_status']}  |  PERCLOS: {alarm_card['perclos_pct']:.1f}%", (20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        cv2.putText(right, f"Yawning: {alarm_card['yawn_status']}  |  Head Tilt: {alarm_card['tilt_status']}  |  Duration: {alarm_card['duration_sec']:.1f}s", (20, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        cv2.putText(right, f"Reason: {alarm_card['reason_summary'][:48]}", (20, y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
        y += 125

        # Section B: Temporal Behavior Graph (WHEN did confidence rise?)
        cv2.putText(right, "WHEN CONFIDENCE ROSE (Temporal Timeline)", (15, y), cv2.FONT_HERSHEY_DUPLEX, 0.50, (255, 255, 255), 1)
        y += 15

        # Plot bounding box
        graph_w = panel_w - 60
        graph_h = 75
        gx = 40
        gy = y + 10
        cv2.rectangle(right, (gx, gy), (gx + graph_w, gy + graph_h), (50, 50, 50), 1)
        cv2.line(right, (gx, gy + int(graph_h * 0.4)), (gx + graph_w, gy + int(graph_h * 0.4)), (80, 80, 80), 1) # 0.6 threshold line

        probs = temporal_data["drowsiness_probabilities"]
        timeline = temporal_data["timeline_seconds"]
        n_pts = len(probs)
        plot_pts = []

        for i, p in enumerate(probs):
            px = gx + int(i * (graph_w / max(1, n_pts - 1)))
            py = gy + graph_h - int(p * graph_h)
            plot_pts.append((px, py))

        for i in range(len(plot_pts) - 1):
            cv2.line(right, plot_pts[i], plot_pts[i + 1], (0, 220, 255), 2)
            cv2.circle(right, plot_pts[i], 3, (0, 255, 0), -1)

        # Axis annotations
        cv2.putText(right, "1.0", (12, gy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
        cv2.putText(right, "0.5", (12, gy + int(graph_h * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
        cv2.putText(right, "0.0", (12, gy + graph_h), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
        cv2.putText(right, f"0.0s", (gx, gy + graph_h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
        cv2.putText(right, f"{timeline[-1]:.1f}s", (gx + graph_w - 25, gy + graph_h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
        
        y += graph_h + 40

        # Section C: Regional SHAP Attributions
        cv2.putText(right, "WHY (Regional Feature Attributions):", (15, y), cv2.FONT_HERSHEY_DUPLEX, 0.50, (255, 255, 255), 1)
        y += 18
        for reg_name, pct in shap_percentages.items():
            b_len = int((panel_w - 160) * (pct / 100.0))
            cv2.putText(right, f"{reg_name[:11]}:", (15, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
            cv2.rectangle(right, (105, y), (105 + b_len, y + 12), (0, 160, 255), -1)
            cv2.putText(right, f"{pct:.1f}%", (115 + b_len, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
            y += 20

        canvas[:, w:] = right
        return canvas
