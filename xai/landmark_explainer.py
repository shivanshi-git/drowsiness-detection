import numpy as np
import cv2


class LandmarkExplainer:
    """
    Facial Landmark & Geometric State Explainer.
    Computes Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and Head Nod/Tilt
    to provide human-interpretable validation of AI drowsiness decisions.
    """
    def __init__(self, ear_threshold: float = 0.22, mar_threshold: float = 0.55):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold

    def calculate_ear(self, eye_pts: np.ndarray) -> float:
        """
        Eye Aspect Ratio (EAR) = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        """
        if len(eye_pts) < 6:
            # Fallback heuristic for bounding box: h / w
            return 0.3
        
        # Vertical distances
        v1 = np.linalg.norm(eye_pts[1] - eye_pts[5])
        v2 = np.linalg.norm(eye_pts[2] - eye_pts[4])
        # Horizontal distance
        h = np.linalg.norm(eye_pts[0] - eye_pts[3])

        ear = (v1 + v2) / max(1e-5, (2.0 * h))
        return float(ear)

    def calculate_mar(self, mouth_pts: np.ndarray) -> float:
        """
        Mouth Aspect Ratio (MAR) = (||p2 - p8|| + ||p3 - p7|| + ||p4 - p6||) / (2 * ||p1 - p5||)
        """
        if len(mouth_pts) < 8:
            return 0.2

        v1 = np.linalg.norm(mouth_pts[1] - mouth_pts[7])
        v2 = np.linalg.norm(mouth_pts[2] - mouth_pts[6])
        v3 = np.linalg.norm(mouth_pts[3] - mouth_pts[5])
        h = np.linalg.norm(mouth_pts[0] - mouth_pts[4])

        mar = (v1 + v2 + v3) / max(1e-5, (2.0 * h))
        return float(mar)

    def explain_landmarks(self, landmarks_dict: dict) -> dict:
        """
        Generates interpretable rule-based fatigue metrics from landmark geometry.
        """
        left_eye_ear = landmarks_dict.get("left_ear", 0.28)
        right_eye_ear = landmarks_dict.get("right_ear", 0.28)
        avg_ear = (left_eye_ear + right_eye_ear) / 2.0
        mar = landmarks_dict.get("mar", 0.25)
        head_pitch = landmarks_dict.get("pitch", 0.0)

        is_eyes_closed = avg_ear < self.ear_threshold
        is_yawning = mar > self.mar_threshold
        is_nodding = head_pitch > 20.0 or head_pitch < -20.0

        reasons = []
        if is_eyes_closed:
            reasons.append(f"Eye Closure: EAR={avg_ear:.2f} < {self.ear_threshold:.2f}")
        if is_yawning:
            reasons.append(f"Yawn Detected: MAR={mar:.2f} > {self.mar_threshold:.2f}")
        if is_nodding:
            reasons.append(f"Head Nodding: Pitch={head_pitch:.1f}°")

        return {
            "avg_ear": avg_ear,
            "mar": mar,
            "head_pitch": head_pitch,
            "is_eyes_closed": is_eyes_closed,
            "is_yawning": is_yawning,
            "is_nodding": is_nodding,
            "explanation_summary": " | ".join(reasons) if reasons else "Facial geometry normal"
        }
