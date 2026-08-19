import cv2
import numpy as np


class VisualizerHUD:
    """
    Renders bounding boxes, facial landmarks, fatigue gauges, and status banners onto video frames.
    """
    def __init__(self):
        pass

    def draw_hud(self, frame_bgr: np.ndarray, alarm_card: dict, fps: float = 30.0) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        canvas = frame_bgr.copy()

        level = alarm_card.get("alarm_level", 0)
        color = (0, 0, 255) if level == 3 else ((0, 140, 255) if level == 2 else (0, 255, 0))

        # Top banner
        cv2.rectangle(canvas, (0, 0), (w, 60), (20, 20, 20), -1)
        prob = alarm_card.get("drowsiness_prob_pct", 0.0)
        cv2.putText(canvas, f"FATIGUE: {prob:.1f}% | FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        cv2.putText(canvas, f"PERCLOS: {alarm_card.get('perclos_pct', 0.0):.1f}% | Reason: {alarm_card.get('reason_summary', '')[:35]}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Alarm border
        if level == 3:
            cv2.rectangle(canvas, (0, 0), (w, h), (0, 0, 255), 6)

        return canvas
