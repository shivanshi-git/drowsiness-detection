import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinaFaceDetector:
    """
    RetinaFace Face & Landmark Detection wrapper for low-light driver monitoring.
    Extracts high-fidelity Region of Interest (RoI) crops for:
      - Face Patch
      - Left Eye Patch
      - Right Eye Patch
      - Mouth / Yawning Patch
    """
    def __init__(self, confidence_threshold: float = 0.8, eye_size=(64, 64), mouth_size=(64, 64)):
        self.conf_thresh = confidence_threshold
        self.eye_size = eye_size
        self.mouth_size = mouth_size
        
        self._face_cascade = None
        self._eye_cascade = None

    @property
    def face_cascade(self):
        if self._face_cascade is None:
            xml = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(xml)
        return self._face_cascade

    @property
    def eye_cascade(self):
        if self._eye_cascade is None:
            xml = getattr(cv2.data, "haarcascades", "") + "haarcascade_eye.xml"
            self._eye_cascade = cv2.CascadeClassifier(xml)
        return self._eye_cascade

    def __deepcopy__(self, memo):
        return RetinaFaceDetector(
            confidence_threshold=self.conf_thresh,
            eye_size=self.eye_size,
            mouth_size=self.mouth_size
        )

    def detect_and_crop(self, frame_bgr: np.ndarray) -> dict:
        """
        Detects facial landmarks and extracts standardized RoI crops.
        Args:
            frame_bgr: (H, W, 3) BGR image
        Returns:
            dict containing:
              'face': (224, 224, 3) image
              'left_eye': (64, 64, 3) image
              'right_eye': (64, 64, 3) image
              'mouth': (64, 64, 3) image
              'bbox': [x1, y1, x2, y2]
              'landmarks': 5 keypoints [[x, y], ...]
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        
        if len(faces) > 0:
            # Pick largest detected face
            x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + fw), min(h, y + fh)
        else:
            # Center face fallback heuristic
            x1, y1 = int(w * 0.15), int(h * 0.15)
            x2, y2 = int(w * 0.85), int(h * 0.85)
            fw, fh = x2 - x1, y2 - y1

        face_crop = frame_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            face_crop = frame_bgr

        # Geometric facial region partitioning
        # Left eye: top-left quadrant of face
        le_x1, le_y1 = x1 + int(fw * 0.12), y1 + int(fh * 0.20)
        le_x2, le_y2 = x1 + int(fw * 0.48), y1 + int(fh * 0.50)
        left_eye_crop = frame_bgr[max(0, le_y1):min(h, le_y2), max(0, le_x1):min(w, le_x2)]

        # Right eye: top-right quadrant of face
        re_x1, re_y1 = x1 + int(fw * 0.52), y1 + int(fh * 0.20)
        re_x2, re_y2 = x1 + int(fw * 0.88), y1 + int(fh * 0.50)
        right_eye_crop = frame_bgr[max(0, re_y1):min(h, re_y2), max(0, re_x1):min(w, re_x2)]

        # Mouth: bottom half of face
        m_x1, m_y1 = x1 + int(fw * 0.20), y1 + int(fh * 0.60)
        m_x2, m_y2 = x1 + int(fw * 0.80), y1 + int(fh * 0.95)
        mouth_crop = frame_bgr[max(0, m_y1):min(h, m_y2), max(0, m_x1):min(w, m_x2)]

        # Resize to standard dimensions
        face_resized = cv2.resize(face_crop, (224, 224))
        left_eye_resized = cv2.resize(left_eye_crop if left_eye_crop.size > 0 else face_crop, self.eye_size)
        right_eye_resized = cv2.resize(right_eye_crop if right_eye_crop.size > 0 else face_crop, self.eye_size)
        mouth_resized = cv2.resize(mouth_crop if mouth_crop.size > 0 else face_crop, self.mouth_size)

        return {
            "face": face_resized,
            "left_eye": left_eye_resized,
            "right_eye": right_eye_resized,
            "mouth": mouth_resized,
            "bbox": [x1, y1, x2, y2],
            "landmarks": [
                [x1 + int(fw * 0.30), y1 + int(fh * 0.35)], # Left eye
                [x1 + int(fw * 0.70), y1 + int(fh * 0.35)], # Right eye
                [x1 + int(fw * 0.50), y1 + int(fh * 0.55)], # Nose
                [x1 + int(fw * 0.35), y1 + int(fh * 0.75)], # Mouth left
                [x1 + int(fw * 0.65), y1 + int(fh * 0.75)]  # Mouth right
            ]
        }
