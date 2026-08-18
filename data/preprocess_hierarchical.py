import os
import glob
import cv2
import numpy as np
import mediapipe as mp
import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def apply_clahe(img_bgr):
    """Applies CLAHE contrast equalization on L-channel in LAB color space."""
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

def extract_group_id(filepath):
    """
    Extracts subject/group identifier to enforce zero data leakage.
    """
    path_obj = Path(filepath)
    filename = path_obj.name.lower()
    full_path_str = str(path_obj).lower()

    # Pattern 1: NTHU-DDD (e.g., 001_glasses_...)
    nthu_match = re.match(r'^(\d{3})_', filename)
    if nthu_match:
        return f"nthu_{nthu_match.group(1)}"

    # Pattern 2: MRL Eye Dataset (e.g., s0013_...)
    mrl_match = re.search(r'(s\d{3,4})', filename)
    if mrl_match:
        return f"mrl_{mrl_match.group(1)}"

    # Pattern 3: Subject / Participant pattern
    subj_match = re.search(r'(subject[_\-]?\d+|participant[_\-]?\d+|sub[_\-]?\d+|\d{2}_\d{2})', full_path_str)
    if subj_match:
        return subj_match.group(0)

    return f"dir_{path_obj.parent.name}"

class HierarchicalDataPreprocessor:
    """
    Hierarchical Data Preprocessor:
    1. Dataset 1 (Eye State Dataset): MRL + Eye crops (128x128) -> ['0_open', '1_closed']
    2. Dataset 2 (Face Drowsiness Dataset): Kaggle/NTHU/UTA Face crops (224x224) -> ['0_alert', '1_drowsy', '2_yawning']
    """
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}

    def __init__(self, eye_size=(128, 128), face_size=(224, 224)):
        self.eye_size = eye_size
        self.face_size = face_size

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
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception:
            self.cascade = None

        self.LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE = [362, 263, 387, 386, 385, 373, 374, 380]

    def extract_face_crop_224(self, frame_bgr):
        """
        Extracts 224x224 full-face crop preserving eyes, mouth, nose, and head posture.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w, _ = frame_bgr.shape

        # Tier 1: MediaPipe Face Mesh Bounding Box
        if self.face_mesh is not None:
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    pts = np.array([(int(l.x * w), int(l.y * h)) for l in landmarks])
                    x_min, y_min = np.min(pts, axis=0)
                    x_max, y_max = np.max(pts, axis=0)

                    # Add 20% margin around face
                    fw = x_max - x_min
                    fh = y_max - y_min
                    pad_x = int(fw * 0.2)
                    pad_y = int(fh * 0.2)

                    x1 = max(0, x_min - pad_x)
                    y1 = max(0, y_min - pad_y)
                    x2 = min(w, x_max + pad_x)
                    y2 = min(h, y_max + pad_y)

                    face_crop = frame_bgr[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        resized = cv2.resize(face_crop, self.face_size)
                        return apply_clahe(resized)
            except Exception:
                pass

        # Tier 2: OpenCV Cascade Face Detection
        if self.cascade is not None and not self.cascade.empty():
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    fx, fy, fw, fh = faces[0]
                    x1 = max(0, fx - int(fw * 0.1))
                    y1 = max(0, fy - int(fh * 0.1))
                    x2 = min(w, fx + fw + int(fw * 0.1))
                    y2 = min(h, fy + fh + int(fh * 0.1))
                    face_crop = frame_bgr[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        resized = cv2.resize(face_crop, self.face_size)
                        return apply_clahe(resized)
            except Exception:
                pass

        # Tier 3: Direct Resize Fallback
        resized = cv2.resize(frame_bgr, self.face_size)
        return apply_clahe(resized)

    def extract_eye_crop_128(self, frame_bgr):
        """
        Extracts 128x128 eye crop specifically for Eye State classification.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w, _ = frame_bgr.shape

        if w < 160 and h < 160:
            resized = cv2.resize(frame_bgr, self.eye_size)
            return apply_clahe(resized)

        if self.face_mesh is not None:
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    pts = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in self.LEFT_EYE])
                    x, y, ew, eh = cv2.boundingRect(pts)
                    pad_x = int(ew * 0.3)
                    pad_y = int(eh * 0.3)
                    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                    x2, y2 = min(w, x + ew + pad_x), min(h, y + eh + pad_y)
                    eye_crop = frame_bgr[y1:y2, x1:x2]
                    if eye_crop.size > 0:
                        resized = cv2.resize(eye_crop, self.eye_size)
                        return apply_clahe(resized)
            except Exception:
                pass

        resized = cv2.resize(frame_bgr, self.eye_size)
        return apply_clahe(resized)

if __name__ == "__main__":
    preprocessor = HierarchicalDataPreprocessor()
    print("[✓] Hierarchical Data Preprocessor initialized successfully.")
