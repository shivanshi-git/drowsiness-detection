import os
import glob
import cv2
import numpy as np
import mediapipe as mp
import re
from pathlib import Path
from tqdm import tqdm

def apply_clahe(img_bgr):
    """Applies CLAHE on L-channel in LAB color space."""
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    try:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        merged = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception:
        return img_bgr

class DualROIProcessor:
    """
    Extracts Dual-ROI: Eye Crop (128x128) + Mouth Yawn Crop (128x128)
    and combines them into a side-by-side 256x128 composite ROI image.
    """
    def __init__(self, single_roi_size=(128, 128)):
        self.single_roi_size = single_roi_size
        self.output_size = (single_roi_size[0] * 2, single_roi_size[1]) # 256x128

        solutions = getattr(mp, 'solutions', None)
        self.face_mesh = None
        if solutions is not None and hasattr(solutions, 'face_mesh'):
            try:
                self.face_mesh = solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                )
            except Exception:
                self.face_mesh = None

        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                self.face_cascade = None
        except Exception:
            self.face_cascade = None

        # Landmark indices
        self.EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153, 362, 263, 387, 386, 385, 373, 374, 380]
        self.MOUTH_INDICES = [61, 81, 13, 311, 291, 402, 14, 178, 0, 17, 37, 267, 84, 314]

    def extract_dual_roi(self, frame_bgr):
        """
        Extracts Eye ROI and Mouth ROI, resizes each to 128x128, and concatenates horizontally (256x128).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w, _ = frame_bgr.shape
        eye_crop = None
        mouth_crop = None

        # 1. MediaPipe Face Mesh Extraction
        if self.face_mesh is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Eye bounding box
                eye_pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.EYE_INDICES])
                ex_min, ey_min = np.min(eye_pts, axis=0)
                ex_max, ey_max = np.max(eye_pts, axis=0)
                pad_e = int(max(ex_max - ex_min, ey_max - ey_min) * 0.3)
                ey1, ey2 = max(0, ey_min - pad_e), min(h, ey_max + pad_e)
                ex1, ex2 = max(0, ex_min - pad_e), min(w, ex_max + pad_e)
                if ey2 > ey1 and ex2 > ex1:
                    eye_crop = cv2.resize(frame_bgr[ey1:ey2, ex1:ex2], self.single_roi_size)

                # Mouth bounding box
                mouth_pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.MOUTH_INDICES])
                mx_min, my_min = np.min(mouth_pts, axis=0)
                mx_max, my_max = np.max(mouth_pts, axis=0)
                pad_m = int(max(mx_max - mx_min, my_max - my_min) * 0.3)
                my1, my2 = max(0, my_min - pad_m), min(h, my_max + pad_m)
                mx1, mx2 = max(0, mx_min - pad_m), min(w, mx_max + pad_m)
                if my2 > my1 and mx2 > mx1:
                    mouth_crop = cv2.resize(frame_bgr[my1:my2, mx1:mx2], self.single_roi_size)

        # Fallback to OpenCV Cascade if MediaPipe landmarks failed
        if (eye_crop is None or mouth_crop is None) and self.face_cascade is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                if eye_crop is None:
                    eye_roi = frame_bgr[fy:fy + int(fh * 0.5), fx:fx + fw]
                    eye_crop = cv2.resize(eye_roi, self.single_roi_size) if eye_roi.size > 0 else None
                if mouth_crop is None:
                    mouth_roi = frame_bgr[fy + int(fh * 0.5):fy + fh, fx:fx + fw]
                    mouth_crop = cv2.resize(mouth_roi, self.single_roi_size) if mouth_roi.size > 0 else None

        # Ultimate fallback: crop upper half (eye) & lower half (mouth)
        if eye_crop is None:
            eye_crop = cv2.resize(frame_bgr[:h // 2, :], self.single_roi_size)
        if mouth_crop is None:
            mouth_crop = cv2.resize(frame_bgr[h // 2:, :], self.single_roi_size)

        # CLAHE contrast equalization
        eye_crop = apply_clahe(eye_crop)
        mouth_crop = apply_clahe(mouth_crop)

        # Side-by-side concatenation: [Eye Crop | Mouth Crop] -> (128, 256, 3)
        dual_roi_composite = np.hstack([eye_crop, mouth_crop])
        return dual_roi_composite

if __name__ == "__main__":
    processor = DualROIProcessor()
    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(dummy_img, (50, 50), (250, 250), (255, 255, 255), -1)
    res = processor.extract_dual_roi(dummy_img)
    print(f"[✓] DualROIProcessor test output shape: {res.shape}")
