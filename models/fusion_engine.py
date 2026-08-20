import torch
import numpy as np

class HierarchicalFusionEngine:
    """
    Hierarchical Fusion Engine:
    Combines Eye State Model (Open/Closed), Face Drowsiness Model (Alert/Drowsy/Yawn),
    and Geometric EAR & MAR indicators into a unified, defensible Drowsiness Score (0.0 to 1.0).
    """
    def __init__(self, eye_weight=0.45, face_weight=0.35, geom_weight=0.20):
        self.eye_weight = eye_weight
        self.face_weight = face_weight
        self.geom_weight = geom_weight

    def evaluate(self, p_eye_closed, p_face_drowsy, p_yawn, ear, mar):
        """
        Evaluates final drowsiness probability and produces evidence dictionary.

        Args:
            p_eye_closed (float): Probability of eye closure from EyeStateModel (0.0 to 1.0)
            p_face_drowsy (float): Probability of facial fatigue from FaceDrowsinessModel
            p_yawn (float): Probability of yawning from FaceDrowsinessModel
            ear (float): Eye Aspect Ratio from MediaPipe FaceMesh (<0.20 indicates closure)
            mar (float): Mouth Aspect Ratio from MediaPipe FaceMesh (>0.50 indicates yawn)

        Returns:
            drowsiness_score (float): Final blended drowsiness score (0.0 to 1.0)
            is_drowsy (bool): Binary decision flag
            evidence (dict): Detailed evidence dictionary for XAI reporting
        """
        # Geometric Signals
        is_ear_closed = (ear < 0.21)
        is_mar_yawning = (mar > 0.52)

        geom_drowsy_prob = 0.0
        if is_ear_closed and is_mar_yawning:
            geom_drowsy_prob = 0.95
        elif is_ear_closed:
            geom_drowsy_prob = 0.75
        elif is_mar_yawning:
            geom_drowsy_prob = 0.65
        else:
            geom_drowsy_prob = 0.05

        # Face fatigue score combines general facial drowsiness and yawning
        face_fatigue_score = max(p_face_drowsy, p_yawn * 0.85)

        # Weighted Fusion
        fusion_score = (
            self.eye_weight * p_eye_closed +
            self.face_weight * face_fatigue_score +
            self.geom_weight * geom_drowsy_prob
        )

        is_drowsy = fusion_score >= 0.50 or (is_ear_closed and p_eye_closed > 0.60)

        evidence = {
            "eye_state": "CLOSED" if p_eye_closed >= 0.50 or is_ear_closed else "OPEN",
            "p_eye_closed": float(p_eye_closed),
            "face_expression": "YAWNING" if p_yawn > 0.40 or is_mar_yawning else ("DROWSY" if p_face_drowsy > 0.50 else "ALERT"),
            "p_face_drowsy": float(p_face_drowsy),
            "p_yawn": float(p_yawn),
            "ear": float(ear),
            "mar": float(mar),
            "is_ear_closed": is_ear_closed,
            "is_mar_yawning": is_mar_yawning,
            "fusion_score": float(fusion_score),
            "verdict": "DROWSY" if is_drowsy else "ALERT"
        }

        return fusion_score, is_drowsy, evidence
